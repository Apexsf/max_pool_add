python generate_test_case.py -s small

cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=OFF 
make 
cd ..
./build/mpa ./test_case 
# ./build/mpa ./test_case >> test_record.txt


cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=ON
make 
cd ..
./build/mpa ./test_case 

cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=OFF
make 
cd ..
./build/mpa ./test_case 



cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE  -DUSE_OMP=ON -DUSE_AVX=ON
make 
cd ..
./build/mpa ./test_case 





python generate_test_case.py -s mid

cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=OFF
make 
cd ..
./build/mpa ./test_case 

cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=ON
make 
cd ..
./build/mpa ./test_case 


cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=OFF
make 
cd ..
./build/mpa ./test_case 


cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=ON
make 
cd ..
./build/mpa ./test_case 








python generate_test_case.py -s large

cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=OFF -DUSE_AVX=OFF
make 
cd ..
./build/mpa ./test_case 

cd build
cmake ..-DCMAKE_BUILD_TYPE=RELEASE  -DUSE_OMP=OFF -DUSE_AVX=ON
make 
cd ..
./build/mpa ./test_case 


cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=OFF
make 
cd ..
./build/mpa ./test_case 


cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DUSE_OMP=ON -DUSE_AVX=ON
make 
cd ..
./build/mpa ./test_case 
