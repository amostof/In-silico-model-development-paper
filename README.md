# In silico model development and optimization of in vitro lung cell population growth

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)

It is authored by Amirmahdi Mostofinejad.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "In-silico-model-development-paper"
```
which auto-activate the project and enable local path handling from DrWatson.

# Cite this work
If you use this code for academic research, you are encouraged to cite the following paper:
If you use this code for academic research, you are encouraged to cite the following paper:
```
@article{Mostofinejad2024InSilico,
    doi = {10.1371/journal.pone.0300902},
    author = {Mostofinejad, Amirmahdi AND Romero, David A. AND Brinson, Dana AND Marin-Araujo, Alba E. AND Bazylak, Aimy AND Waddell, Thomas K. AND Haykal, Siba AND Karoubi, Golnaz AND Amon, Cristina H.},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {In silico model development and optimization of in vitro lung cell population growth},
    year = {2024},
    month = {05},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0300902},
    pages = {1-27},
    number = {5},
}
```

## Questions

To get help on how to use the code, simply open an issue in the GitHub "Issues" section.
