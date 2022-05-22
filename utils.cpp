
#include "utils.hpp"
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <string.h>
#include <fstream>


std::vector<std::string> list_dir(const char *path) {
    struct dirent *entry;
    DIR *dir = opendir(path);
    std::vector<std::string> paths;
    if (dir == NULL) {
        return paths;
    }
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0 
        && strcmp(entry->d_name, "..")!=0)
        paths.emplace_back(entry->d_name);
    }
    closedir(dir);
    return paths;
}

 int get_file_size(const char* path){
    FILE *f;
    f = fopen(path , "r");
    fseek(f, 0, SEEK_END);
    size_t len = (unsigned long)ftell(f);
    fclose(f);
    return len;
 }






