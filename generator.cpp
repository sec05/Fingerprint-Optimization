#include "generator.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <random>

using namespace OPT;
using namespace LAMMPS_NS;

Generator::Generator(char* f){
    inputFile = f;
}

Generator::~Generator(){
   delete calibrator;
}

NLA::Matrix* Generator::generate_fingerprint_matrix(){
    numRadialFingerprints = 2;
    numBondFingerprints = 10;
    radialFingerprintsLowerBound = 0;
    radialFingerprintsUpperBound = 1;
    bondFingerprintsLowerBound = 0;
    bondFingerprintsUpperBound = 1;
    // create input file
    generate_opt_inputs();
    std::cout << "generated input!"<<std::endl;
    // then pass it through set FOR NOW
    inputFile += ".opt";
    std::string temp = "./Optimizer Output/"+inputFile; 
    char* f = new char[temp.size()+1];
    std::strcpy(f,temp.c_str());
    calibrator = new PairRANN(f);
    calibrator->setup();
   // need to find the matrix size to allocate **FIND BETTER METHOD**
    int rows = 0;
    int columns = 0;
    for(int n = 0; n < calibrator->nsims; n++){
        LAMMPS_NS::PairRANN::Simulation& sim = calibrator->sims[n];
        for(int i = 0; i < sim.inum; i++){
            rows++;
            columns = calibrator->net[calibrator->map[sim.type[sim.ilist[i]]]].dimensions[0];
        }
    }
    // write matrix to file
    // create matrix ******* CHECK SIZE FOR BIGGER INPUTS
    std::ofstream matrixFile;
    NLA::Matrix* m = new NLA::Matrix(rows,columns);
    int r = 0;
    matrixFile.open("./Optimizer Output/"+inputFile+".matrix");
    for(int n = 0; n < calibrator->nsims; n++){
        LAMMPS_NS::PairRANN::Simulation& sim = calibrator->sims[n];
        for(int i = 0; i < sim.inum; i++){
            for(int j = 0; j < columns; j++){
                matrixFile << sim.features[i][j];
                std::string delimerator = j == columns - 1 ? "" : ", ";
                matrixFile << delimerator;
                m->data[r][j] = sim.features[i][j];
            }
            r++;
            matrixFile << std::endl;
        }
    }
    matrixFile.close();
    printf("Generator: created fingerprint matrix!\n");
    return m;
}

void Generator::generate_opt_inputs()
{
    // want to read every line from input file into
    std::map<int, std::pair<std::string, std::string> > *inputTable = readFile();
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
    if (atomTypes.size() != 1)
    {
        printf("Generator error: only supports one atom systems!\n");
        return;
    }
    #ifdef DEBUG
    printf("DEBUG: Found atom types:");
    for(std::string a : atomTypes) printf("%s ",a.c_str());
    printf("\n");
    #endif
    // make sure number of fingerprints is number we are going to generate
    int fnIndex = utils::searchTable("fingerprintsperelement:"+atomTypes[0]+":", inputTable);
    inputTable->at(fnIndex).second = std::to_string(numRadialFingerprints+1);
    #ifdef DEBUG
    printf("DEBUG: fingerprintsperelement=%d\n",numRadialFingerprints+1);
    #endif

    // find amount of alphas ie compute |o - n| + 1 assuming o < 0 and n > 0
    // make sure they are in the map by looking for key then get value
    // first find index in ordered map
    int oIndex = utils::searchTable("fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:o:", inputTable);
    // if we cant find it then error
    if (oIndex == -1)
    {
        printf("Generator error: could not find %s in input file!\n", ("fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:o:").c_str());
        return;
    }
    // then we take first number given
    int o = std::stoi(utils::splitString((*inputTable)[oIndex].second).at(0));

    // same thing for n's
    int nIndex = utils::searchTable("fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:n:", inputTable);
    if (nIndex == -1)
    {
        printf("Generator error: could not find %s in input file!\n", ("fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:n:").c_str());
        return;
    }
    int n = std::stoi(utils::splitString((*inputTable)[nIndex].second).at(0));

    // now we can calculate alphas

    // if o or either n are bellow 0 then we abs and add 1 otherwise we just abs
    int alphas = 0;
    if ((o < 0 && n > 0) || (o > 0 && n < 0))
        alphas = abs(o - n) + 1;
    else
        alphas = abs(o - n);
    #ifdef DEBUG
    printf("DEBUG: Found o=%d and n=%d setting alphas=%d\n",o,n,alphas);
    printf("DEBUG: Generating radial template\n");
    #endif
    // we now need to make a "template" of the radial blocks
    // first we assemble all the keys we are looking for
    std::vector<std::string> radialKeys = {"fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:re:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:rc:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:alpha:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:dr:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:o:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":radialscreened_0:n:"};
    std::vector<std::string> radialValues = {};

    // now we interate through the list and create our template values
    for (std::string key : radialKeys)
    {
        int index = utils::searchTable(key, inputTable);
        if (index == -1)
        {
            printf("Generator error: radial: cannot find %s in input file!\n", key.c_str());
            return;
        }
        radialValues.push_back((*inputTable)[index].second);
    }
    #ifdef DEBUG
    printf("DEBUG: Generating bond template\n");
    #endif
    // we now do the same thing for bond
    std::vector<std::string> bondKeys{"fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":bondscreened_0:re:","fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":bondscreened_0:rc:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":bondscreened_0:alphak:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":bondscreened_0:dr:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":bondscreened_0:k:", "fingerprintconstants:" + atomTypes.at(0) + "_" + atomTypes.at(0) + "_" + atomTypes.at(0) + ":bondscreened_0:m:"};
    std::vector<std::string> bondValues = {};

    // now we interate through the list and create our template values
    for (std::string key : bondKeys)
    {
        int index = utils::searchTable(key, inputTable);
        if (index == -1)
        {
            printf("Generator error: bond; cannot find %s in input file!\n", key.c_str());
            return;
        }
        bondValues.push_back((*inputTable)[index].second);
    }

    // need to first find indicies of m and k
    int m = -1;
    int alpha_kIndex = -1;
    for(int i = 0; i < bondKeys.size(); i++){
        size_t pos = bondKeys.at(i).find(":m:");
        if(pos != std::string::npos){
            m = atoi(bondValues.at(i).c_str());
        }
        pos = bondKeys.at(i).find(":k:");
        if(pos != std::string::npos){
            bondValues.at(i) = std::to_string(numBondFingerprints);
        }
        pos = bondKeys.at(i).find("alphak");
        if(pos != std::string::npos) alpha_kIndex = i;
    }
    if(m == -1){
        printf("Generator error: could not find m!\n");
        return;
    }
    if(alpha_kIndex == -1){
        printf("Generator error: could not find alpha_k!\n");
        return;
    }
    #ifdef DEBUG
    printf("DEBUG: Finished generating templates \n");
    #endif

    // need to update the layer 0 size
    int layer0Index = utils::searchTable("layersize:"+atomTypes[0]+":0:",inputTable);
    inputTable->at(layer0Index).second = std::to_string(m*numBondFingerprints+alphas*numRadialFingerprints);
    #ifdef DEBUG
    printf("DEBUG: Finished updating layer 0 size \n");
    #endif

    // before adding on fingerprints we must get rid of the existing ones
    int elements = 0;
   for (auto it = inputTable->begin(); it != inputTable->end(); ) {
        std::string key = it->second.first;
        if (key.find("fingerprintconstants:") != std::string::npos) {
            it = inputTable->erase(it);  // erase returns the next iterator
        } else {
            ++it;
        }
        elements++;
    }
    #ifdef DEBUG
    printf("DEBUG: Finished removing old fingerprints\n");
    #endif
    // we now write the amount of randialscreened_0,1,2,3...
    int rIndex = utils::searchTable("fingerprints:"+atomTypes[0]+"_"+atomTypes[0]+":",inputTable);
    std::string radials = "";
    for(int i = 0; i < numRadialFingerprints; i++){
        std::string temp = "radialscreened_"+std::to_string(i);
        temp += i != numRadialFingerprints - 1 ? " " : ""; 
        radials += temp;
    }
    inputTable->at(rIndex).second = radials;
    
    // we now write to entire map back to a new file to save space
    std::filesystem::create_directories("./Optimizer Output");
    std::ofstream fingerprintsFile;
    fingerprintsFile.open("./Optimizer Output/"+inputFile +".opt");
    if(!fingerprintsFile.is_open()){
        printf("Generator error: cannot create optimizer input file!\n");
        return;
    }
    for(const auto& entry : *inputTable)
    {
        fingerprintsFile<<entry.second.first<<std::endl;
        fingerprintsFile<<entry.second.second<<std::endl;
    }

    delete inputTable;
    #ifdef DEBUG
    printf("DEBUG: Finished writing map back to file and deleting table\n");
    #endif
    // open file to write generated columns
    std::ofstream fingerprintsVectorFile;
    fingerprintsVectorFile.open("./Optimizer Output/"+inputFile+".optv");
    if(!fingerprintsVectorFile.is_open()){
        printf("Generator error: cannot create fingerprint vector file!\n");
        return;
    }

    // now we add on new blocks of radial fingerprints
    // set up random generation
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(radialFingerprintsLowerBound,radialFingerprintsUpperBound);
    
    // loop and do generation for radial
    for(int i = 0; i < numRadialFingerprints; i++){
        // generate alphas and write them to the vector
        std::string generatedAlphas;
        for(int j = o; j <= n; j++){
            double alpha = dist(gen);
            generatedAlphas += std::to_string(alpha);
            generatedAlphas += j==n ? "" : " ";
            fingerprintsVectorFile << "n="<<j<<", alpha="<<alpha<<std::endl;
        }
        //update values in radial keys
        for(int j = 0; j < radialKeys.size(); j++){
            size_t pos = radialKeys.at(j).find("radialscreened_"+std::to_string(i-1));
            if(pos != std::string::npos){
                radialKeys.at(j).replace(pos,16,"radialscreened_"+std::to_string(i));
            }
            // update with new alphas if needed
            pos = radialKeys.at(j).find("alpha");
            if(pos != std::string::npos){
                radialValues.at(j) = generatedAlphas;
            }
        }

        //write radial fingerprints to file
        for(int j = 0; j < radialKeys.size(); j++){
            fingerprintsFile << radialKeys.at(j) << std::endl;
            fingerprintsFile << radialValues.at(j) << std::endl;
        }
    }
    #ifdef DEBUG
    printf("DEBUG: Finished writing new radial fingerprints\n");
    #endif
    //write to vector and optimized input file
    dist = std::uniform_real_distribution<double>(bondFingerprintsLowerBound,bondFingerprintsUpperBound);
    std::string alpha_ks = "";
    for(int k = 1; k <= numBondFingerprints; k++)
    {
        
        double alpha_k = dist(gen);
        alpha_ks += std::to_string(alpha_k);
        alpha_ks += k == numBondFingerprints ? "" : " ";
        for(int j = 0; j < m; j++){
        fingerprintsVectorFile << "k="<<k<<", m="<<j<<", alpha_k="<<alpha_k<<std::endl;
        }
    }
    bondValues.at(alpha_kIndex) = alpha_ks;

    for(int i = 0; i < bondKeys.size(); i++){
        fingerprintsFile << bondKeys.at(i) << std::endl;
        fingerprintsFile << bondValues.at(i) << std::endl;
    }
    #ifdef DEBUG
    printf("DEBUG: Finished writing new bond fingerprints\n");
    #endif
    fingerprintsFile.close();
    fingerprintsVectorFile.close();

    // std::filesystem::copy_file(inputFile, strcat(inputFile,".optimizer"));
}

// function for parsing top comment line with generation requirements
void Generator::parse_parameters(){
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
std::map<int, std::pair<std::string, std::string> > *Generator::readFile()
{
    printf("Generator: reading input file!\n");
    std::map<int, std::pair<std::string, std::string> > *pairs = new std::map<int, std::pair<std::string, std::string> >();

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