#include "SampleRNN.h"

// Initialize the static members of the SampleRNN class
map<char, int> SampleRNN::_ctoi;
map<int, char> SampleRNN::_itoc;
unique_ptr<Mat> SampleRNN::_one_hots;