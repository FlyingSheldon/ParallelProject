echo "start building pp"
cd ..
cmake -S . -DCMAKE_BUILD_TYPE:STRING=Release -B build/
cmake --build build
echo "finish building pp"
cd stage
mkdir -p bin/
cp ../build/src/pp bin/pp