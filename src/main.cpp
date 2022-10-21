#include "nana/basic_types.hpp"
#include <ctime>
#include <iostream>
#include <nana/gui.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/label.hpp>
#include <random>

#include "sim.h"

#ifdef CUDA_AVAL
#include "cudamatmul.h"
#endif

using namespace nana;

void launch_sim(form& fm)
{
    Grid g(400, 400);
    Transmitter tx { 0, 0, 22, 2400 };

    OSM(g, tx, 2);
    // g.print();

    double g_max = g.get_max_val();
    double g_min = g.get_min_val();
    double range = g_max - g_min;

    drawing dw(fm);
    dw.draw([g, g_max, g_min, range](paint::graphics& graph) {
        // graph.rectangle(rectangle { 5, 5, 800, 600 }, true, colors::white);
        graph.rectangle(rectangle { 0, 0, g.size_x, g.size_y }, true, colors::white);
        for (uint32_t x = 0; x < g.size_x; x++) {
            for (uint32_t y = 0; y < g.size_y; y++) {
                nana::color c;
                double r_val = (((g.get_val(x, y) - g_min) * 255) / range);
                c.from_rgb(r_val, r_val, r_val);
                graph.set_pixel(x, y, c);
            }
        }
    });
    dw.update();
}

int main(void)
{
    srand(time(NULL));
#ifdef CUDA_AVAL
    cudaMul();
#endif

    form fm;
    launch_sim(fm);

    fm.events().click(API::exit_all);

    fm.show();

    // //Start to event loop process, it blocks until the form is closed.
    exec();

    return 0;
}