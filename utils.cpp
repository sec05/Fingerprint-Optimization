#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>

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