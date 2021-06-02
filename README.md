# intphys-renderer

- ```Recurrent-Interaction-Network```: Is an edited version of https://github.com/rronan/Recurrent-Interaction-Network.  Main changes are to cnr/model.py,
to make it work as a isolated module, without the provided wrapper.  The inputs expected and the forward function
were also changed to allow dataparallel to work prooperly.  It should not require further edits to run the scripts as they are.

```Recurrent-Interaction-Network``` also contains ```pyronan_git``` which is a git repository taken from https://github.com/rronan/pyronan and ```pyronan``` 
which is the actual library from this repository.  

- Running the script ```train_render.py``` should be sufficient to start training a model with default parameters.  Running with the -h flag provides help messages for some of the command line parameters. Currently the script uses my weights and biases login, which will have to be changed when someone else runs it.  This script contains the parser parameters as well as the training loop for the compositional renderer using the intphys dataset.  Currently it is very messy and lacks proper commmenting which I will try to fix.  Some parts of thhe scrips should also be set as commandline arguments, instead of setting them in the middle of the script like I do, which I can als try to fix.

- The script ```json_render_dataloader.py``` contains the dataset and related code for the renderer training.  Lines 45-59 handle removing undesirable training examples,
such as scenes without visble objects, scenes with walls and scenes with occluders.  Currently too change what actally gets removed I just edit the 
conditional on line 55, though this should probably be handled differently, sorry for the bad practice.



