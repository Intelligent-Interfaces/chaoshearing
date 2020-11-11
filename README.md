# Chaos Hearing

 _[Erick Oduniyi](https://eoduniyi.github.io/erick.light/) & [Myunghyun Oh](https://mathematics.ku.edu/myunghyun-oh)_

## [On Listening and Duets](): _Hearing Performance, Chaotic Dynamics, and Computational Auditory Scenery Analysis_

<div style="text-align: justify">
This first project ("On Listening and Duets") and the entire Chaos Hearing project have been started because of a deep appreciation of mathematics, music, and sound. The Chaos Hearing project's goal is to facilitate the conversations between these entities (sound, music, and math. These conversations are between student and teacher, artist and artist, mathematician and mathematician, and artist and mathematician. During the conversations, expect discussions on related research, theories, art, design, and everything else from a formal and wonderfully artistic perspective.
</div>

<!-- ---
## Examples of Theory

We use `latex` to specify mathematical formulations:

```latex
% The normal form equation for the supercritical Hopf bifurication w/ AWGN: n_z(t)
\begin{align}
	\dfrac{dz(t)}{dt} = (\mu i\omega_0)z(t) - (a + i/3)|z(t)|^2z(t) + n_{z}(t)
\end{align}
```

$$
\dfrac{dz(t)}{dt} = (\mu i\omega_0)z(t) - (a + i/3)|z(t)|^2z(t) + n_{z}(t)
$$

---
## Examples of Code

We use `python`, `matlab`, `R`, `swift`, and various other programming languages for implementations of theory.

[### Deep Learning in Swift (Iris ANN, The Hello World of DL)]()

> The Iris genus entails about **300 species**, but our program will only classify the following three:
>
> * Iris setosa
> * Iris virginica
> * Iris versicolor

<table>
  <tr><td>
    <img src="https://www.tensorflow.org/images/iris_three_species.jpg"
         alt="Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor">
  </td></tr>
  <tr><td align="center">
    <b>Figure 1.</b> <a href="https://commons.wikimedia.org/w/index.php?curid=170298">Iris setosa</a> (by <a href="https://commons.wikimedia.org/wiki/User:Radomil">Radomil</a>, CC BY-SA 3.0), <a href="https://commons.wikimedia.org/w/index.php?curid=248095">Iris versicolor</a>, (by <a href="https://commons.wikimedia.org/wiki/User:Dlanglois">Dlanglois</a>, CC BY-SA 3.0), and <a href="https://www.flickr.com/photos/33397993@N05/3352169862">Iris virginica</a> (by <a href="https://www.flickr.com/photos/33397993@N05">Frank Mayfield</a>, CC BY-SA 2.0).<br/>
  </td></tr>
</table>


#### 🧠 Creating a neural network model to classify flowers 

<table>
  <tr><td>
    <img src="https://www.tensorflow.org/images/custom_estimators/full_network.png"
         alt="A diagram of the network architecture: Inputs, 2 hidden layers, and outputs">
  </td></tr>
  <tr><td align="center">
    <b>Figure 2.</b> A fully-connected neural network consisting of an input layer (features), two hidden layers, and an output layer (predictions).<br/>
  </td></tr>
</table>

```swift
// Import Swift for TensorFlow Deep Learning Library
import Tensorflow
import Pythonkit
import Python

/** Define a neural network
  We'll define a neural network that is three layers:layer1 = features -> size(hidden_nodes)
  layer2 = size(hidden_nodes) -> size(hidden_nodes)
  layer3 = size(hidden_nodes) -> p(features) = {labels}
**/

// Set the # of artificial neurons
let hiddenSize: Int = 10

// Create a model: fully-connected Neural Network (NN)
struct lightIrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
		
    // Differentiable Programming! 
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}
```

[Reinforcement Learning in Swift (Cart-Pole, The Classic Control Example)]()

---
Contact Us
#####  <i>Erick Oduniyi ([eeoduniyi@gmail](eeoduniyi@gmail.com))</i>      
 -->
