#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <limits>
#include <set>
#include <map>

#include "Mat.h"
#include "SampleRNN.h"
#include "BasicRNN.h"
#include "LstmRNN.h"

extern "C" {
#include <cblas.h>
}

using namespace std;

vector<string> read_lines(set<char>& unique_chars, int max_lines = -1) {
    // ifstream file("data/names.txt");
    ifstream file("data/tinyshakespeare.txt");

    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        exit(1);
    }

    vector<string> lines;
    string line;
    int cur_line = 0;

    while (getline(file, line)) {
        if (max_lines != -1 && cur_line >= max_lines)
            break;
        if (line.length() > 0 && line.back() != ':') {
            for (char c : line)
                unique_chars.insert(c);
            lines.push_back("_" + line);
            cur_line++;
        }
    }
    
    file.close();

    return lines;
}


int main(int argc, char const *argv[]) {

    std::srand(std::time(0));

    goto_set_num_threads(2);
    openblas_set_num_threads(2);
    
    int max_seq_len = 25;
    int token_space_size = 256;

    set<char> unique_chars;

    vector<string> sequences = read_lines(unique_chars);

    size_t max_length = 0;
    size_t min_length = numeric_limits<size_t>::max();
    size_t total_length = 0;

    for (const auto& seq : sequences) {
        size_t length = seq.length();
        max_length = max(max_length, length);
        min_length = min(min_length, length);
        total_length += length;
    }

    double mean_length = static_cast<double>(total_length) / sequences.size();

    cout << "Sequences count: " << sequences.size() << endl;
    cout << "Longest sequence length: " << max_length << endl;
    cout << "Mean sequence length: " << mean_length << endl;
    cout << "Shortest sequence length: " << min_length << endl;

    cout << "Unique characters: " << unique_chars.size() << endl;

    vector<char> characters(unique_chars.begin(), unique_chars.end());
    characters.insert(characters.begin(), '_'); // Special character for padding

    map<char, int> char_to_idx;
    for (size_t i = 0; i < characters.size(); i++)
        char_to_idx[characters[i]] = i;

    map<int, char> idx_to_char;
    for (size_t i = 0; i < characters.size(); i++)
        idx_to_char[i] = characters[i];

    SampleRNN::set_ctoi_and_itoc(char_to_idx, idx_to_char);

    SampleRNN* samples[sequences.size()];

    for (size_t i = 0; i < sequences.size(); i++) {
        samples[i] = new SampleRNN(sequences[i], max_seq_len);
    }

    // BasicRNN rnn(characters.size(), 100, characters.size());
    LstmRNN rnn(characters.size(), 50, characters.size());
    rnn.initialize_params();

    vector<string> generated = rnn.generate('_', max_seq_len, 10);

    for (const string& s : generated) {
        cout << s << endl;
    }


    cout << "sequences.size() = " << sequences.size() << endl;

    float learning_rate = 0.1;
    int epochs = 10;
    int mini_batch_size = 50;

    rnn.sgd(samples, sequences.size(), learning_rate, epochs, mini_batch_size);

    cout << "Training complete." << endl;

    generated = rnn.generate('_', max_seq_len, 30);

    for (const string& s : generated) {
        cout << s << endl;
    }

    return 0;
}