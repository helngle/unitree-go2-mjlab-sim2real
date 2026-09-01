#include <yaml-cpp/yaml.h>

#include <vector>

#include "isaaclab/manager/manager_term_cfg.h"


int main()
{
    isaaclab::ObservationTermCfg term;
    term.history_length = 3;
    term.scale = {1.0f, 1.0f};
    term.reset({1.0f, 2.0f});
    if (term.get() != std::vector<float>({1, 2, 1, 2, 1, 2})) return 1;
    term.add({3.0f, 4.0f});
    term.add({5.0f, 6.0f});
    if (term.get() != std::vector<float>({1, 2, 3, 4, 5, 6})) return 2;
    term.add({7.0f, 8.0f});
    if (term.get() != std::vector<float>({3, 4, 5, 6, 7, 8})) return 3;
    return 0;
}
