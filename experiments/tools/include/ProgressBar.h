#ifndef MAT_SUBSET_EXPERIMENTS_PROGRESS_BAR_H
#define MAT_SUBSET_EXPERIMENTS_PROGRESS_BAR_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace MatSubset::Experiments {

class ProgressBar {
  public:
    ProgressBar(size_t total, size_t bar_width = 50)
        : total(total), bar_width(bar_width), current(0),
          start_time(std::chrono::steady_clock::now()) {}

    void update(const std::string &label = "") {
        current++;
        display(label);
    }

    void set_progress(size_t value, const std::string &label = "") {
        current = value;
        display(label);
    }

    void finish(const std::string &message = "Done!") {
        current = total;
        display("");
        std::cout << " " << message << std::endl;
    }

  private:
    size_t total;
    size_t bar_width;
    size_t current;
    std::chrono::steady_clock::time_point start_time;

    void display(const std::string &label) {
        double progress = static_cast<double>(current) / total;
        size_t filled = static_cast<size_t>(progress * bar_width);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - start_time)
                           .count();

        // Build the progress bar with shade blocks for finer granularity
        // Unicode shade characters: ░ (light) ▒ (medium) ▓ (dark) █ (full)
        const char* blocks[] = {" ", "░", "▒", "▓", "█"};

        double progress_chars = progress * bar_width;
        size_t full_blocks = static_cast<size_t>(progress_chars);
        double fractional_part = progress_chars - full_blocks;
        size_t partial_block_index = static_cast<size_t>(fractional_part * 4);

        std::ostringstream bar;
        bar << "\033[1A\r\033[K";  // Move up 1 line, return to start, clear line
        if (!label.empty()) {
            bar << label << " ";
        }
        bar << "▕";
        for (size_t i = 0; i < bar_width; ++i) {
            if (i < full_blocks) {
                bar << "█";
            } else if (i == full_blocks && partial_block_index > 0) {
                bar << blocks[partial_block_index];
            } else {
                bar << " ";
            }
        }
        bar << "▏" << std::setw(3) << static_cast<int>(progress * 100)
            << "% [" << current << "/" << total << "] " << elapsed << "s";

        // Pad with spaces to clear previous output and add newline below
        bar << "    \n";

        std::cout << bar.str() << std::flush;
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_PROGRESS_BAR_H
