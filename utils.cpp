#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "omp.h"
#include "utils.h"

using namespace LAMMPS_NS;

  double utils::numeric(const char *,int,std::string line,bool,char *){return atof(line.c_str());}
  double utils::inumeric(const char *,int,std::string line,bool,char *){return atoi(line.c_str());}
  std::string utils::trim_comment(const std::string &line){
      auto end = line.find("#");
      if (end != std::string::npos) { return line.substr(0,end); }
      return line;  
  }
  FILE * utils::open_potential(char *filename,char *,std::nullptr_t){return fopen(filename,"r");}
  char * utils::strdup(const std::string &text){
      auto tmp = new char[text.size() + 1];
      strcpy(tmp, text.c_str());
      return tmp;
  }

// helper function to turn spaces between strings into words
std::vector<std::string> utils::splitString(const std::string& str) {
    std::vector<std::string> result;
    std::istringstream iss(str);
    std::string token;
    
    while (iss >> token) {
        result.push_back(token);
    }
    
    return result;
}

// helper function to search ordered map
int utils::searchTable(std::string key, std::map<int, std::pair<std::string, std::string> >* table){
    // iterate through the pairs until key matches then return index in table
    for (const auto& index : *table) {
        if(index.second.first == key) return index.first;
    }
    return -1;
}

// string helpers taken from stack overflow
std::string utils::trim(const std::string& str, const std::string& whitespace = " \t")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

std::string utils::reduce(const std::string& str,const std::string& fill = " ",const std::string& whitespace = " \t")
{
    // trim first
    auto result = trim(str, whitespace);

    // replace sub ranges
    auto beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const auto endSpace = result.find_first_not_of(whitespace, beginSpace);
        const auto range = endSpace - beginSpace;

        result.replace(beginSpace, range, fill);

        const auto newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}