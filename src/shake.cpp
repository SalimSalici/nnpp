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

using namespace std;

vector<string> read_lines(set<char>& unique_chars, int max_lines = -1) {
    // Create and open a text file
    ifstream file("data/names.txt");

    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        exit(1);
    }

    vector<string> lines;
    string line;
    int cur_line = 0;

    // Read from the text file
    while (getline(file, line)) {
        if (max_lines != -1 && cur_line >= max_lines)
            break;
        if (line.length() > 0 && line.back() != ':') {
            for (char c : line)
                unique_chars.insert(c);
            lines.push_back(line);
            cur_line++;
        }
    }
    
    // Close the file
    file.close();

    return lines;
}

int main(int argc, char const *argv[]) {
    
    int mini_batch_size = 3;
    int max_seq_len = 15;
    int token_space_size = 256;

    set<char> unique_chars;

    vector<string> sequences = read_lines(unique_chars);

    // Calculate sequence statistics
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

    // Print the results   

    cout << "Sequences count: " << sequences.size() << endl;
    cout << "Longest sequence length: " << max_length << endl;
    cout << "Mean sequence length: " << mean_length << endl;
    cout << "Shortest sequence length: " << min_length << endl;

    cout << "Unique characters: " << unique_chars.size() << endl;
    // for (char c : unique_chars)
    //     cout << c << " ";

    vector<char> characters(unique_chars.begin(), unique_chars.end());
    characters.insert(characters.begin(), '_'); // Special character for padding

    map<char, int> char_to_idx;
    for (size_t i = 0; i < characters.size(); i++)
        char_to_idx[characters[i]] = i;

    map<int, char> idx_to_char;
    for (size_t i = 0; i < characters.size(); i++)
        idx_to_char[i] = characters[i];

    SampleRNN::set_ctoi_and_itoc(char_to_idx, idx_to_char);

    cout << "Seq[19]: " << sequences[19] << endl;

    for (size_t i = 0; i < 20; i++) {
        SampleRNN s(sequences[i], max_seq_len);
        cout << SampleRNN::one_hots_to_string(s.get_input_seq_one_hots()) << endl;
        cout << SampleRNN::one_hots_to_string(s.get_output_seq_one_hots()) << endl;
        cout << "++++++++++++++++++" << endl;
    }

    return 0;
}