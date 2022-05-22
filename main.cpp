#include "max_pool_add.hpp"
#include "utils.hpp"
#include <chrono>




using  milliseconds = std::chrono:: milliseconds;
using hclock = std::chrono::high_resolution_clock;


template<typename T>
bool test_from_torch(const char* path_a, 
const char* path_b, const char* path_c, int case_index, 
std::string& data_type,  milliseconds& overall_time) {

    Tensor<T> a(path_a); // construct tensor from file
    Tensor<T> b(path_b); // construct tensor from file
    Tensor<T> c(path_c); // construct tensor from file

    auto t_start = hclock::now(); 
    Tensor<T> res  = max_pool_add(a, b);
    auto t_end = hclock::now();
    overall_time += std::chrono::duration_cast< milliseconds>(t_end - t_start ); 

    bool is_pass = (c == res);
    std::cout << "test case " << case_index << ",\t\t"
    << "a: " << a.B << " x " << a.C << " x "  << a.H << " x " << a.W << ",\t\t"
    << "b: " << b.B << " x " << b.C << " x "  << b.H << " x " << b.W << ",\t\t"
    << "data type: " << data_type<< ",\t\t"
    << (is_pass? " passed": " not passed") << std::endl;
    return is_pass;
}

void test_from_torch(const char * folder_path) { 

    auto get_case_index_from_name = [](std::string str) -> int {
            int first = str.find("_");
            int second  = str.find("_",first+1);
            std::string digit_str = str.substr(first+1, second - first - 1);
            int case_index = std::stoi(digit_str);
            return case_index;
    };

    auto get_data_type_from_name = [](std::string str)->std::string {
            int first = str.find("_");
            int second  = str.find("_",first+1);
            int third = str.find("_", second + 1);
            return str.substr(second+1, third - second -1);
    };


    std::vector<std::string> paths = list_dir(folder_path);
    std::sort(paths.begin(),paths.end(),[get_case_index_from_name](std::string a, std::string b) -> int{ //sort cases by index
       return get_case_index_from_name(a) < get_case_index_from_name(b);
    });

    std::string path_prefix = folder_path + std::string("/case_");
    bool is_pass;
    size_t pass_cnt = 0;
    size_t case_nums = paths.size() / 3;
     milliseconds overall_time(0);
    for (int i = 0; i < case_nums ; i += 1) {
        std::string data_type = get_data_type_from_name(paths[i*3]);
        std::string path_a = path_prefix +
        std::to_string(i) + "_" + data_type + std::string("_a.bin"); // path of tensor a

        std::string path_b = path_prefix +
        std::to_string(i) + "_" + data_type  + std::string("_b.bin"); // path of tensor b

        std::string path_res = path_prefix +
        std::to_string(i) + "_" + data_type + std::string("_c.bin"); // path of tensor c

        if (data_type == "int32") {
            is_pass = test_from_torch<int>(path_a.c_str(), path_b.c_str(), path_res.c_str(),i, data_type, overall_time);
        } else if (data_type == "float32"){
            is_pass = test_from_torch<float>(path_a.c_str(), path_b.c_str(), path_res.c_str(),i, data_type, overall_time);
        } else if (data_type == "double"){
            is_pass = test_from_torch<double>(path_a.c_str(), path_b.c_str(), path_res.c_str(),i, data_type, overall_time);
        } else {
            std::cerr << "Not supported data type" << std::endl;
            abort();
        }
  
        if (is_pass) pass_cnt++;
     
    }

    std::cout << "all cases : " << case_nums << ", passed cases : " << pass_cnt << std::endl;
    std::cout << "overall running time: " <<overall_time.count()<<"  milliseconds\n";

}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Please specify a path to the directory of test cases" << std::endl;
        exit(1);
    } else{
        test_from_torch(argv[1]);
    }

    return 0;
}