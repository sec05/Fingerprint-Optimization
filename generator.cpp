#include "generator.h"
#include "utils.h"
//#include "omp.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <random>

using namespace OPT;
using namespace LAMMPS_NS;

Generator::Generator(char *f)
{
    inputFile = f;
}

Generator::~Generator()
{
    delete calibrator;
}

std::vector<arma::dmat *> Generator::generate_fingerprint_matrix(int numRadialFingerprints, double radialFingerprintsLowerBound, double radialFingerprintsUpperBound, int numBondFingerprints, double bondFingerprintsLowerBound, double bondFingerprintsUpperBound)
{
    // create input file
    this->numRadialFingerprints = numRadialFingerprints;
    this->numBondFingerprints = numBondFingerprints;
    this->radialFingerprintsLowerBound = radialFingerprintsLowerBound;
    this->radialFingerprintsUpperBound = radialFingerprintsUpperBound;
    this->bondFingerprintsLowerBound = bondFingerprintsLowerBound;
    this->bondFingerprintsUpperBound = bondFingerprintsUpperBound;
    generate_opt_inputs();
    std::cout << "generated input!" << std::endl;
    // then pass it through set FOR NOW
    inputFile += ".opt";
    std::string temp = "./Optimizer Output/" + inputFile;
    char *f = new char[temp.size() + 1];
    std::strcpy(f, temp.c_str());
    calibrator = new PairRANN(f);
    calibrator->setup();
    calibrator->normalize_data();
    // need to find the matrix size to allocate **FIND BETTER METHOD**
    int networks = calibrator->nelements;

    // create a list of rows x cols for each matrix
    std::vector<int> rows(networks, 0);
    std::vector<int> columns(networks, 0);
    
    for (int n = 0; n < calibrator->nsims; n++)
    {
        LAMMPS_NS::PairRANN::Simulation &sim = calibrator->sims[n];
        for (int i = 0; i < sim.inum; i++)
        {
            // find how big each matrix is
            int index = calibrator->map[sim.type[sim.ilist[i]]];
            rows.at(index)++;
            columns.at(index) = calibrator->net[index].dimensions[0];
        }
    }

    //calibrator->normalize_data();

    // create matrices
    std::vector<arma::dmat *> matrices;
    for (int i = 0; i < rows.size(); i++)
    {
        matrices.push_back(new arma::dmat(rows.at(i), columns.at(i)));
    }

    std::vector<int> cursor(rows.size(), 0);
    for (int n = 0; n < calibrator->nsims; n++)
    {
        LAMMPS_NS::PairRANN::Simulation &sim = calibrator->sims[n];
        for (int i = 0; i < sim.inum; i++)
        {
            int index = calibrator->map[sim.type[sim.ilist[i]]];
            int cols = columns.at(index);
            for (int j = 0; j < cols; j++)
            {
                matrices.at(index)->at(cursor.at(index), j) = sim.features[i][j];
            }
            cursor.at(index)++;
        }
    }

    delete calibrator->sims;
    delete calibrator->net;
    delete[] f;
    return matrices;
}

void Generator::generate_opt_inputs()
{
    // want to read every line from input file into
    std::map<int, std::pair<std::string, std::string>> *inputTable = readFile();
    if (inputTable == NULL)
    {
        printf("Generator error: could not generate new inout file");
    }

    // read atom types
    int atomTypesIndex = utils::searchTable("atomtypes:", inputTable);
    if (atomTypesIndex == -1)
    {
        printf("Generator error: atom types not found in input file!\n");
        return;
    }
    std::vector<std::string> atomTypes = utils::splitString((*inputTable)[atomTypesIndex].second);
    printf("Generator: found %zu atom types!\n", atomTypes.size());
    //create/clean atom types .optv files
    for(std::string atom : atomTypes){
        std::ofstream f;
        f.open("./Optimizer Output/"+inputFile+"."+atom+".optv");
        f.close();
    }
    // make sure number of fingerprints is number we are going to generate for each atom
    int numAtoms = atomTypes.size();
    for (std::string atom : atomTypes)
    {
        int fnIndex = utils::searchTable("fingerprintsperelement:" + atom + ":", inputTable);
        inputTable->at(fnIndex).second = std::to_string(numRadialFingerprints * numAtoms + numAtoms);
    }

    // need to generate all combinations of atoms for radial
    // example atoms = [Ti,Ni] we make Ti_Ti, Ti_Ni, Ni_Ni, Ni_Ti
    std::vector<std::string> radialCombinations;
    for (int i = 0; i < atomTypes.size(); i++)
    {
        for (int j = 0; j < atomTypes.size(); j++)
        {
            radialCombinations.push_back(atomTypes[i] + "_" + atomTypes[j]);
        }
    }

    // do the same thing for bond fingerprints
    std::vector<std::string> bondCombinations;
    for (int i = 0; i < atomTypes.size(); i++)
    {
        for (int j = 0; j < atomTypes.size(); j++)
        {
            if (atomTypes[i] == atomTypes[j])
                bondCombinations.push_back(atomTypes[i] + "_" + atomTypes[j] + "_" + atomTypes[i]);
            else
                bondCombinations.push_back(atomTypes[i] + "_" + atomTypes[j] + "_all");
        }
    }
    // we now find the amount of alphas for each radial combination
    std::vector<int> alphas, os, ns; // same length as radial combinations
    for (std::string combination : radialCombinations)
    {
        // find amount of alphas ie compute |o - n| + 1 assuming o < 0 and n > 0
        // make sure they are in the map by looking for key then get value
        // first find index in ordered map
        int oIndex = utils::searchTable("fingerprintconstants:" + combination + ":radialscreened_0:o:", inputTable);
        // if we cant find it then error
        if (oIndex == -1)
        {
            printf("Generator error: could not find %s in input file!\n", ("fingerprintconstants:" + combination + ":radialscreened_0:o:").c_str());
            return;
        }
        // then we take first number given
        int o = std::stoi(utils::splitString((*inputTable)[oIndex].second).at(0));
        os.push_back(o);

        // same thing for n's
        int nIndex = utils::searchTable("fingerprintconstants:" + combination + ":radialscreened_0:n:", inputTable);
        if (nIndex == -1)
        {
            printf("Generator error: could not find %s in input file!\n", ("fingerprintconstants:" + combination + ":radialscreened_0:n:").c_str());
            return;
        }
        int n = std::stoi(utils::splitString((*inputTable)[nIndex].second).at(0));
        ns.push_back(n);

        // now we can calculate alphas

        // if o or either n are bellow 0 then we abs and add 1 otherwise we just abs
        int alpha = 0;
        if ((o < 0 && n > 0) || (o > 0 && n < 0))
            alpha = abs(o - n) + 1;
        else
            alpha = abs(o - n);

        alphas.push_back(alpha);
    }

    // we need a template for each radial combination so we create a list of templates
    // remember to free!!!
    std::vector<std::vector<std::string> *> radialCombinationTemplateKeys;
    std::vector<std::vector<std::string> *> radialCombinationTemplateValues;
    for (std::string combination : radialCombinations)
    {
        // we now need to make a "template" of the radial blocks
        // first we assemble all the keys we are looking for
        std::vector<std::string> *radialKeys = new std::vector<std::string>({"fingerprintconstants:" + combination + ":radialscreened_0:re:", "fingerprintconstants:" + combination + ":radialscreened_0:rc:", "fingerprintconstants:" + combination + ":radialscreened_0:alpha:", "fingerprintconstants:" + combination + ":radialscreened_0:dr:", "fingerprintconstants:" + combination + ":radialscreened_0:o:", "fingerprintconstants:" + combination + ":radialscreened_0:n:"});
        std::vector<std::string> *radialValues = new std::vector<std::string>();

        // now we interate through the list and create our template values
        for (std::string key : *radialKeys)
        {
            int index = utils::searchTable(key, inputTable);
            if (index == -1)
            {
                printf("Generator error: radial: cannot find %s in input file!\n", key.c_str());
                return;
            }
            radialValues->push_back((*inputTable)[index].second);
        }

        radialCombinationTemplateKeys.push_back(radialKeys);
        radialCombinationTemplateValues.push_back(radialValues);
    }

    // we now do the same thing for bond
    std::vector<std::vector<std::string> *> bondCombinationTemplateKeys;
    std::vector<std::vector<std::string> *> bondCombinationTemplateValues;

    std::vector<int> alpha_kIndices;

    for (std::string combination : bondCombinations)
    {
        std::vector<std::string> *bondKeys = new std::vector<std::string>({"fingerprintconstants:" + combination + ":bondscreened_0:re:", "fingerprintconstants:" + combination + ":bondscreened_0:rc:", "fingerprintconstants:" + combination + ":bondscreened_0:alphak:", "fingerprintconstants:" + combination + ":bondscreened_0:dr:", "fingerprintconstants:" + combination + ":bondscreened_0:k:", "fingerprintconstants:" + combination + ":bondscreened_0:m:"});
        std::vector<std::string> *bondValues = new std::vector<std::string>();

        // now we interate through the list and create our template values
        for (std::string key : *bondKeys)
        {
            int index = utils::searchTable(key, inputTable);
            if (index == -1)
            {
                printf("Generator error: bond; cannot find %s in input file!\n", key.c_str());
                return;
            }
            bondValues->push_back((*inputTable)[index].second);
        }

        // while we are doing this iteration we should update the value of k, find the value for m, and save
        // the index of alpha_k so we can easily generate new values for it later
        // need to first find indicies of m and k
        int m = -1;
        int alpha_kIndex = -1;

        for (int i = 0; i < bondKeys->size(); i++)
        {
            size_t pos = bondKeys->at(i).find(":m:");
            if (pos != std::string::npos)
            {
                m = atoi(bondValues->at(i).c_str());
            }
            pos = bondKeys->at(i).find(":k:");
            if (pos != std::string::npos)
            {
                bondValues->at(i) = std::to_string(numBondFingerprints);
            }
            pos = bondKeys->at(i).find("alphak");
            if (pos != std::string::npos)
                alpha_kIndex = i;
        }
        if (m == -1)
        {
            printf("Generator error: could not find m!\n");
            exit(1);
        }
        if (alpha_kIndex == -1)
        {
            printf("Generator error: could not find alpha_k!\n");
            exit(1);
        }
        ms.push_back(m);
        alpha_kIndices.push_back(alpha_kIndex);
        bondCombinationTemplateKeys.push_back(bondKeys);
        bondCombinationTemplateValues.push_back(bondValues);
    }

    totalRadial = std::vector<int>(atomTypes.size());
    totalBond = std::vector<int>(atomTypes.size());

    // need to update the layer 0 size
    int counter = 0;
    for (std::string atom : atomTypes)
    {
        int layer0Index = utils::searchTable("layersize:" + atom + ":0:", inputTable);
        int size = 0;

        // iterate through combinations finding the amount of input for each atom type 
        for (int i = 0; i < radialCombinations.size(); i++)
        { 
            std::string combinationType = radialCombinations.at(i);
            combinationType = combinationType.substr(0, combinationType.find_first_of('_'));
            if (combinationType == atom)
            {
                size += alphas.at(i) * numRadialFingerprints;
                totalRadial.at(counter) +=  alphas.at(i) * numRadialFingerprints;
            }
        }

        for (int i = 0; i < bondCombinations.size(); i++)
        {
            std::string combinationType = bondCombinations.at(i);
            combinationType = combinationType.substr(0, combinationType.find_first_of('_'));
            if (combinationType == atom)
            {
                size += ms.at(i) * numBondFingerprints;
                totalBond.at(counter) += ms.at(i) * numBondFingerprints;
            }
        }
        counter++;
        inputTable->at(layer0Index).second = std::to_string(size);
    }

    // before adding on fingerprints we must get rid of the existing ones
    int elements = 0;
    for (auto it = inputTable->begin(); it != inputTable->end();)
    {
        std::string key = it->second.first;
        if (key.find("fingerprintconstants:") != std::string::npos || key.find("bundle") != std::string::npos)
        {
            it = inputTable->erase(it); // erase returns the next iterator
        }
        else
        {
            ++it;
        }
        elements++;
    }

    // we now write the amount of randialscreened_0,1,2,3... for each combination
    for (std::string combination : radialCombinations)
    {
        int rIndex = utils::searchTable("fingerprints:" + combination + ":", inputTable);
        std::string radials = "";
        for (int i = 0; i < numRadialFingerprints; i++)
        {
            std::string temp = "radialscreened_" + std::to_string(i);
            temp += i != numRadialFingerprints - 1 ? " " : "";
            radials += temp;
        }
        inputTable->at(rIndex).second = radials;
    }

    // we now write entire map back to a new file to save space
    std::filesystem::create_directories("./Optimizer Output");
    std::ofstream fingerprintsFile;
    fingerprintsFile.open("./Optimizer Output/" + inputFile + ".opt");
    if (!fingerprintsFile.is_open())
    {
        printf("Generator error: cannot create optimizer input file!\n");
        exit(1);
    }
    for (const auto &entry : *inputTable)
    {
        fingerprintsFile << entry.second.first << std::endl;
        fingerprintsFile << entry.second.second << std::endl;
    }

    delete inputTable;

    // open file to write generated columns and clean it
    std::ofstream fingerprintsVectorFile;

    // now we add on new blocks of radial fingerprints
    // we know we need to generate for each combination
    for (int ii = 0; ii < radialCombinations.size(); ii++)
    {
        std::string combinationType = radialCombinations.at(ii);
        combinationType = combinationType.substr(0, combinationType.find_first_of('_'));
        fingerprintsVectorFile.open("./Optimizer Output/" + inputFile + "." + combinationType + ".optv",std::ios::app);
        if (!fingerprintsVectorFile.is_open())
        {
            printf("Generator error: cannot create fingerprint vector file!\n");
            exit(1);
        }
        int o = os.at(ii);
        int n = ns.at(ii);
        int alpha = alphas.at(ii);
        std::vector<std::string> radialKeys = *(radialCombinationTemplateKeys.at(ii));
        std::vector<std::string> radialValues = *(radialCombinationTemplateValues.at(ii));

        double step = (radialFingerprintsUpperBound - radialFingerprintsLowerBound) / (numRadialFingerprints * alpha);
        int count = 1;

        // loop and do generation for radial
        for (int i = 0; i < numRadialFingerprints; i++)
        {
            // generate alphas and write them to the vector
            std::string generatedAlphas;
            for (int j = o; j <= n; j++)
            {
                double alpha = radialFingerprintsLowerBound + (count * step);
                generatedAlphas += std::to_string(alpha);
                generatedAlphas += j == n ? "" : " ";
                fingerprintsVectorFile << "n=" << j << ", alpha=" << alpha << std::endl;
                count++;
            }

            // update values in radial keys
            for (int j = 0; j < radialKeys.size(); j++)
            {
                size_t pos = radialKeys.at(j).find("radialscreened_" + std::to_string(i - 1));
                if (pos != std::string::npos)
                {
                    size_t p = radialKeys.at(j).find_last_of(":");
                    p = radialKeys.at(j).find_last_of(":", p - 1);
                    std::string variable = radialKeys.at(j).substr(p);
                    radialKeys.at(j) = "fingerprintconstants:" + radialCombinations.at(ii) + ":radialscreened_" + std::to_string(i) + variable;
                }
                // update with new alphas if needed
                pos = radialKeys.at(j).find("alpha");
                if (pos != std::string::npos)
                {
                    radialValues.at(j) = generatedAlphas;
                }
            }

            // write radial fingerprints to file
            for (int j = 0; j < radialKeys.size(); j++)
            {
                fingerprintsFile << radialKeys.at(j) << std::endl;
                fingerprintsFile << radialValues.at(j) << std::endl;
            }
        }
        fingerprintsVectorFile.close();
    }

    // write to vector and optimized input file
    // we do the same thing for each bond combination
    for (int i = 0; i < bondCombinations.size(); i++)
    {
        std::vector<std::string> bondValues = *(bondCombinationTemplateValues.at(i));
        std::vector<std::string> bondKeys = *(bondCombinationTemplateKeys.at(i));
        int m = ms.at(i);
        int alpha_kIndex = alpha_kIndices.at(i);
        double step = (bondFingerprintsUpperBound - bondFingerprintsLowerBound) / numBondFingerprints;
        std::string alpha_ks = "";

        std::string combinationType = bondCombinations.at(i);
        combinationType = combinationType.substr(0, combinationType.find_first_of('_'));
        fingerprintsVectorFile.open("./Optimizer Output/" + inputFile + "." + combinationType + ".optv",std::ios::app);

        for (int k = 1; k <= numBondFingerprints; k++)
        {

            double alpha_k = bondFingerprintsLowerBound + (k * step);
            alpha_ks += std::to_string(alpha_k);
            alpha_ks += k == numBondFingerprints ? "" : " ";
            for (int j = 0; j < m; j++)
            {
                fingerprintsVectorFile << "k=" << k << ", m=" << j << ", alpha_k=" << alpha_k << std::endl;
            }
        }
        bondValues.at(alpha_kIndex) = alpha_ks;

        for (int i = 0; i < bondKeys.size(); i++)
        {
            fingerprintsFile << bondKeys.at(i) << std::endl;
            fingerprintsFile << bondValues.at(i) << std::endl;
        }
        fingerprintsVectorFile.close();
    }

    fingerprintsFile.close();
    fingerprintsVectorFile.close();

    // delete the template pointers

    for (int i = 0; i < radialCombinationTemplateKeys.size(); i++)
    {
        delete radialCombinationTemplateKeys.at(i);
        delete radialCombinationTemplateValues.at(i);
    }

    for (int i = 0; i < bondCombinationTemplateKeys.size(); i++)
    {
        delete bondCombinationTemplateKeys.at(i);
        delete bondCombinationTemplateValues.at(i);
    }

    // finally we print out the lotal amount of radial and bonds generated for each element
    for(int i  = 0; i < atomTypes.size(); i++){
        printf("%s: Generated %d radial and %d bond fingerprints.\n",atomTypes.at(i).c_str(),totalRadial.at(i),totalBond.at(i));
    }
}

// function for parsing top comment line with generation requirements
void Generator::parse_parameters()
{
    /*std::ifstream f(inputFile);
    if (!f)
    {
        printf("Generator error: cannot  access %s", inputFile.c_str());
        return;
    }
    char c;
    f.get(c);
    if(c != '#'){
        printf("Generator error: cannot find optimization parameters in %s",inputFile.c_str());
    }
    // grab first line
    std::string line;
    std::getline(f, line);
    // loop over an assign values
    std::string token;

    size_t pos = 0;
    while(pos < line.size()){
        token = line.substr(pos,line.find('='));

        if(token == "rf")
    }*/
}

// function for parsing out fingerprint sections from
std::map<int, std::pair<std::string, std::string>> *Generator::readFile()
{
    printf("Generator: reading input file!\n");
    std::map<int, std::pair<std::string, std::string>> *pairs = new std::map<int, std::pair<std::string, std::string>>();

    // open file for reading
    std::ifstream f(inputFile);
    if (!f)
    {
        printf("Generator error: cannot  access %s", inputFile.c_str());
        return NULL;
    }

    // if the first line is comments we skip it
    char c;
    f.get(c);
    if (c == '#')
    {
        while (f.get(c))
        {
            if (c == '\n')
                break;
        }
    }
    // iterate down the file and add pairs to the map
    std::string key, value;
    std::regex pattern("^\\s+|\\s+$");
    int order = 0;
    while (true)
    {
        if (std::getline(f, key))
        {
            if (std::getline(f, value))
            {
                std::regex_replace(key, pattern, "");
                std::regex_replace(value, pattern, "");
                (*pairs)[order] = std::pair<std::string, std::string>(key, value);
                order++;
            }
        }
        else
        {
            break;
        }
    }
    /*
    for (const auto& pair : *pairs) {
         int key = pair.first;
         const auto& value = pair.second;
         std::cout << "Key: " << key << ", Pair: {" << value.first << ", " << value.second << "}" << std::endl;
     }
     */
    f.close();
    return pairs;
}