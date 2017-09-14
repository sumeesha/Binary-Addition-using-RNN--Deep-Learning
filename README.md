# Binary-Addition--Deep-Learning

The topics which i have mentioned are a part of my unfinished study. Please consider reading these topics one by one to get a quick idea of the basics.

<p>
<ul>
<li>Initially we calculate <b>Sigmoid functions</b>. Sigmoid functions are used as the activation function of the neurons.</li>

<li>The<b> activation function</b> of a node decides whether or not the neuron is to be considered, i.e, activated. Read this amazing
<a href="https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0"> article</a> on Activation function. This guy has explained everything we need to know on this topic.

</li>

<li>Next comes <b>Cost function</b>

<b>Cost function</b> – calculates the difference between the desired output and the real output, i.e, the error.
It expresses how wrong our model is. Therefore, lesser cost function value implies better output.

<img src="http://tutorial.math.lamar.edu/Classes/CalcI/BusinessApps_files/image001.gif" />


The required piont is the global <b>mininima</b> or else the nearest point to x.(only one such point exists).
Here x represents the weight. As a result, we gain the best W value for which Cost function is minimum => minimum error => desired output.

Training a network means <b>minimizing</b> the cost function.
<li>
So, How to obtain the optimum weight? In other words, How to plot the above graph?
(The above graph is the representation of Cost fn w.r.t Weight (for 1 neuron). Plotting the graph by Brute Force Method would take 10^27 secs (for n = 1000 weights/neurons) which is longer than the existence of the universe.)
</li>
</li>

<li>The solution is <b>Gradient Descent</b>.
<b>Gradient Descent</b> – Helps to find which way is the <b>downhill</b> in the graph so that we can find the local/global minima (<b>least weight</b>). To speed up things, we use <b>“Derivative”</b>.

<b>Derivative</b> – rate of change of error w.r.t rate of change of weight. (dE/dW) In other words, The slope of the curve.
If  

		 dE/dW   =   +ve, then
		
		the cost function is going uphill (undesired)


	     dE/dW   =   -ve, then
		
		the cost function is going downhill (desired, as it takes us to the minima).


	Finally, if  
			dE/dW  =  0, then minima (desired point). 



<img src="https://github.com/sumeesha/Sentence-Correction--Deep-Learning/blob/master/Screenshot%20from%202017-09-14%2003-34-00.png" />
</li>
 </ul>
 </p>
 
