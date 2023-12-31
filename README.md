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
@unpublished{Mostofinejad2024InSilico,
  title = {{\textit{In silico} model development and optimization of \textit{in vitro} lung cell population growth}},
  author = {Mostofinejad, Amirmahdi and Romero, David A. and Brinson, Dana and Marin-Araujo, Alba E. and Bazylak, Aimy and Waddell, Thomas K. and Haykal, Siba and Karoubi, Golnaz and Amon, Cristina H.},
  note = {Submitted to PLOS One},
  year = {2024}
}
```

## Questions

To get help on how to use the code, simply open an issue in the GitHub "Issues" section.
