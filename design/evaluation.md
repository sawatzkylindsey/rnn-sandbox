### Evaluation
We evaluate the effectiveness of the nn-wd tool by conducting user studies where users must use the tool to diagnose some issue with a NN.
Users are separated into different groups, where each group debugs the NN problems with a different tool technique (the problems are consistent, only the tool/technique changes).
If the nn-wd tool is effective, its users will be able to more quickly & accurrately diagnose issues.
It is possible that some tools will be more effective at diagnosing different classes of problems, so a representative series of problems should be presented.

### Cases
Here we outline the various cases that this study can be run with.
Notice in these cases, we never change the diagramming of the NN computational graph, or its mathematical symbols.
These must remain consistent, otherwise the tasks become trivially solved.

* Wrong operator (switch vector addition for dot product, or visa versa).
* Wrong weight vector (even though the NN learns some contextualized gate, we always set to 1^N or 0^N).
* Mis-trained weights (so even though we state the NN should learn the sentences 'walk the dog' and 'feed the cat', we train the NN for 'walk the cat' and 'feed the dog').
This will be an interesting case, and we want the tool to show that this problem can be diagnosed to a high degree of accurracy.
So if the tool does its job, the user should be able to pinpoint areas on the NN that have learned the wrong weights, and what those weights actually should be.
This is highly idealized, but ultimately would probably solidify the tool's value - we'll see if we can actually get there!

