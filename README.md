# VocalizationAnalysis

To take a look into the method, I advise to use at the **Interactive Notebook**. It's detailing the steps of the methods, using widgets to see the effects of the parameters.

EXPERIMENT

The **experiment** files manage all the go-no-go task, and calls the **RecordingAnalysis4experiment** file to make the analysis.

CONSTRUCTION OF THE METHOD

The **lib0-dep** file analyse presegmented files one by one and save the result in the form of spectrogram plots. Usefull for visualisation of the results.

The **Optimisation** file compute a Bayesian optimisation of the parameters and uses the **RecordingAnalysis4** file.

The **figure for report** notebook creates the figures (and it's quite messy)

TODO
- recheck the files
- some reorganization
- comment
