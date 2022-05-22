python generate_test_case.py -s mid -n 100 -o test_case
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=ON
make
cd ..
./build/mpa ./test_case