cd tests/cpp/
TE_LIB_PATH=$(pip3 show transformer-engine | grep -E "Location:|Editable project location:" | tail -n 1 | awk '{print $NF}')

echo "TE_LIB_PATH:" 
echo $TE_LIB_PATH 

# pause 2 second
sleep 2

cmake -GNinja -Bbuild .
cmake --build build


echo "How to run test?"
echo "cd build"
echo "operator/test_operator --gtest_list_tests"
echo "operator/test_operator --gtest_filter=OperatorTest/FusedCastMXFP8TestSuite.*"


# operator/test_operator --gtest_filter=OperatorTest/NormTestSuite.TestNorm/Te*