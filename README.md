# **Is your AI a people pleaser? A mechanistic overview of language models and sycophancy**

## **Introduction**

LLM’s have now become an essential part of our lives and we not only rely on it for your professional work but also for companionship and therapy. If we trust these models so much, a rational thing to assume would be that we understand the inner workings of these models and how they generate their outputs but the reality is that these LLMs are a black box and we have very little knowledge about their internal workings. This problem led to the upcoming of a new field of research in AI called mechanistic interpretability.
In this blog we will start off with exploring how LLMs actually learn to do different tasks and do they create some kind of internal world representation of the task they are performing. Then we will look at how to reverse engineer LLMs and try to discover some actual circuits. Finally we will look at a real world application of the interpretability in combatting and discovering sycophancy in LLMs 

## Table of Contents

- [Introduction](#introduction)
  - [Preliminaries](#preliminaries)
    - [Residual Stream](#residual-stream)
    - [Virtual Weights](#virtual-weights)
  - [Linear Representation](#linear-representation)
    - [Superposition](#superposition)
- [Grokking](#grokking)
  - [Exploring the Generalised Solution](#exploring-the-generalised-solution)
  - [World Representation](#world-representation)
- [Othello-GPT](#othello-gpt)
  - [Probing](#probing)
  - [Causality of Representation](#causality-of-representation)
  - [Does Othello-GPT Have Linearly Encoded Board States?](#does-othello-gpt-have-linearly-encoded-board-states)
  - [Additional Linear Interpretations](#additional-linear-interpretations)
- [Circuit Discovery](#circuit-discovery)
  - [Indirect Object Identification in GPT-2 Small](#indirect-object-identification-in-gpt-2-small)
  - [Automated Circuit Discovery (ACDC)](#automated-circuit-discovery-acdc)
- [Sycophancy](#sycophancy)
  - [Introduction](#introduction-1)
  - [Delusional Spiralling](#delusional-spiralling)
  - [Mechanistic Account of Sycophancy](#mechanistic-account-of-sycophancy)
- [Conclusion](#conclusion)


## **Preliminaries**
### Residual Stream
In a traditional feed-forward network, each layer is a function that transforms the output of the previous layer. If a layer fails to learn something useful, it affects the signal for every subsequent operation. In a Transformer, we use Residual Connections. Every layer(whether it’s an Attention head or an MLP) does not replace the input, but it adds its result to it. The next layer then performs some operations on the result from this addition and then, in turn, adds its output to that input, which is then “read” by the next layer and so on. This creates a communication channel through which the layers “read” the information, perform some operations on it, and then “write” on it (by adding). This channel is called Residual stream.

<p align="center">
<img width="650" height="650" alt="Screenshot 2026-04-15 at 7 07 06 PM" src="https://github.com/user-attachments/assets/c10afab9-711d-4c14-b8bb-644b01a2d7a9" />
</p>

While a layer's internal computations are non-linear, the Residual stream itself maintains a highly linear structure. Every layer performs a linear operation to “read” from the residual stream. After performing its own operations - which may be non-linear - inside the residual stream, it again performs a linear operation to “write” its output onto the residual stream. A powerful consequence of this linearity is the emergence of Virtual Weight.

### Virtual Weights
Since the residual is linear in structure, we can mathematically skip the stream while seeing the interaction between any two layers. For example, let  be the projection matrix through which layer 1 “writes” into the residual stream, and  Let be the projection matrix through which layer 3 “reads” from the residual stream. Then, multiplying these two weights yields “virtual weights”  connecting layer 1 and layer 3. Now, these virtual weights quantify the amount of “signal” that flows directly from layer 1 to layer 3. Because the stream is additive, this direct connection exists regardless of what is happening inside layer 2.

<p align="center">
<img width="650" height="650" alt="Screenshot 2026-04-15 at 7 20 18 PM" src="https://github.com/user-attachments/assets/280c74bc-daf1-4b75-b2b5-bbed31d44a95" />
</p>

### Linear Representation
It has been observed that many high-level concepts are represented as linear directions inside a model's representation space. By high-level concepts, we mean, for example, if the text is about an object, is it black or white? If it mentions a person, is the person male or female? etc..Because these concepts are linear directions, we can use them to control the model’s behaviors. This is known as model steering. If we identify the vector for a specific concept, we can manually add it to the residual stream during processing. For example, adding a “black” vector can force the sentence to be about a “black sofa” rather than a “white sofa”. Similarly, adding a “female” vector can force a model’s output to be about a “queen” rather than a “king”. This allows us to edit the model's "thoughts" in real-time without retraining its weights. Steering shows that these linear directions are not just correlations but rather causal to the model's decisions.


### Superposition
We know that a space with n dimensions can only hold n perfectly orthogonal vectors. But high-dimensional space has a unique property. In a high-dimensional space, it is possible to have an exp(n) number of almost orthogonal vectors. This allows the model to contain far more concepts than there are dimensions in the residual stream. We call this phenomenon Superposition. The model "packs" high-level concepts into the stream by allowing their directions to overlap slightly. For this packing to work without excessive interference, the features must be sparse. In fact, most of the important concepts the model learns are inherently sparse. These overlaps between concepts make mechanistic interpretability difficult. Because concept directions are not aligned with individual neuron axes, the neurons become polysemantic. This means we cannot look at a single neuron to find a single concept, as that neuron’s activity is likely shared by many different, overlapping directions.

<p align="center">
<img width="324" height="312" alt="image" src="https://github.com/user-attachments/assets/d3fe3586-3de4-4344-bbd6-d71948bd2289" />
</p>

*Figure 3: In superposition, a neuron becomes polysemantic when the number of features exceeds the number of neurons. Because it is impossible to align the feature directions to the neuron axes*	


## **Grokking**

<p align="center">
<img width="986" height="192" alt="image" src="https://github.com/user-attachments/assets/514f003b-48a4-4369-9aa6-3b8ab8c54f2a" />
</p>

*Figure 4: Characteristic grokking loss curve trained on modular addition data, showcasing a clear dip in test loss (red) from 8k epoch to 14k*

How do LLMs learn something new? Do they memorize, or can they find a general solution? Grokking is the phenomenon of  sudden transition from memorization to a general algorithm (Characterized by the sharp fall in test loss fig1)
To understand and isolate this phenomenon, we observe it on a toy task of modular addition (a + b mod p) using a small one-layer transformer (Refer to the appendix for details on the transformer architecture). We observe that in the early stages the training loss decreases sharply(expected as it is an easy task) but the test loss is high and saturates indicating overfitting where the model has memorised all answers where the weights act effectively like a lookup table,but as training continues at one point the test loss drops and becomes close to train loss indicating that the model is suddenly able to perform well on unseen examples hence reached a general algorithm.
The field of mechanistic interpretability is about reverse-engineering models to human interpretable circuits. Hence, we can think of models as having circuits that perform certain tasks. Now, let's imagine our modular addition task has a circuit X that needs to be formed in our model to perform it. Now in the start of the training phase the model prefers memorisation solution as the components which make up the circuit are not yet formed and lined up properly therefore the average of L2 norm of the all weights in such a case is very high as they have to store a lookup table therefore to force the model towards the general solution (Circuit formation) there is a need of regularisation (weight decay) which forces the model towards lower weight norm and hence simpler and general solution. As the training continues, we reach a point where the circuit is completely formed and the testing loss drops dramatically.


## Exploring the generalised solution
Here we explore the generalised algorithm the model was able to learn which turns out to be mapping the inputs onto a circle and performing addition on the circle.

<p align="center">
<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/df8a293b-d521-47ac-9768-83276b229749" />
</p>

Formally the algorithm involves projecting two integers a and b into its corresponding rotation using the embedding matrix. The embedding matrix just like all the other weight matrices in the model only contains some sparse set of key frequencies wk in the fourier domain. The attention and the mlp layers learn to calculate the sine and cosine of a+b and the final unembedding and output projection matrices then calculate the output logit c. Now when we take the softmax of the logits to make our prediction and take the most probable answer what we get is basically c = a + b (mod p). ( cosine(w(a+b-c)) is maximum at a+b-c = 0)

<p align="center">
<img width="600" height="600" alt="clock_animation-3" src="https://github.com/user-attachments/assets/08055a70-2a0e-4154-b4a3-b41fffcff5f9" />
</p>

*Figure 6: Each token (0–112) has a 128-dimensional embedding vector. To visualise all 113 of them at once, we project these vectors onto the two Fourier basis directions at the model's dominant frequency, collapsing 128 dimensions down to 2. The result is a 2D snapshot of how the model internally represents numbers, and watching it over training reveals the moment the model stops memorising and represents the numbers as different points lying on a circle.*


This solution might feel too elegant and simple.You may wonder how do we know the model is performing its computation in the frequency domain? The answer lies in examining different weight matrices. Take the embedding matrix for instance when we apply a Fourier transform along the input dimension then compute the ℓ2-norm along the other dimension and plot them we observe peaks at only few sparse frequencies(wk) this validates that the model is indeed operating in this basis. A memorisation solution would produce an unstructured embedding matrix whose Fourier distribution would spread roughly uniformly across all frequencies resulting in  a flat plot. What we observe instead is the vast majority of energy collapsed into just 3-5 frequencies. Moreover it is observed that these same frequencies are the key frequencies in other matrices as well, further strengthening the claim.

<p align="center">
<img width="788" height="126" alt="Screenshot 2026-04-15 at 7 33 14 PM" src="https://github.com/user-attachments/assets/3db08336-e6ef-4dad-9ee7-db0321d8a044" />
</p>

*Figure 7: The embedding is sparse in the Fourier basis, and throws away all Fourier components apart from a handful of frequencies
Periodic structure occurs in all other model components such as MLP activations, attention scores for different heads as well as the residual stream.They are always periodic with one of the key frequencies that we discovered earlier*

<p align="center">
<img width="305" height="223" alt="Screenshot 2026-04-15 at 7 33 53 PM" src="https://github.com/user-attachments/assets/3241888a-3a85-44b4-846f-8c36be08f0f0" />
</p>
*Figure 8:  Neuron's activation as y sweeps from 0 to 112 with x fixed at 17. Blue is the pre-ReLU activation of a clean sine wave, showing the neuron is computing a pure Fourier feature of the form sin(ω(x+y) + φ). Red is the post-ReLU activation the negative half of the sine is clipped to zero, leaving only the positive peaks*

<p align="center">
<img width="631" height="136" alt="Screenshot 2026-04-15 at 7 34 41 PM" src="https://github.com/user-attachments/assets/875508c8-d160-4ac7-983f-efa5833c3ec5" />
</p>

*Figure 9: Each panel shows one dimension of the 128-dim residual stream at the final token position, as y sweeps from 0 to 112 with x fixed at 17. After grokking, individual dimensions of the residual stream oscillate sinusoidally with y the model has learned to encode the input as a Fourier feature rather than as a raw number.*

Ablation study is used extensively in mechanistic interpretability to ensure that the circuit we found actually has a causal role in the model output and is not just an intermediate artifact with no real computational significance. It involves zeroing out the activations of certain components and measuring the downstream effect on model outputs. Here when we ablate the key frequencies (Figure 5), we see the loss spike up. Ablating any other frequency leaves the loss completely unchanged. This is strong evidence that the model is indeed performing the described algorithm.
Figure 5 The loss of the transformer (lower is better) when we ablate all the frequencies notice the red spikes in loss when we ablate the key frequencies 

Earlier we observed that the model has a point where it ‘groks’ and generalizes (test loss becomes close to the train loss). This process might feel abrupt but when you dig deeper it turns out to be a gradual change from the memorised to the general solution. To understand this we need some progress measures which are computed during training and tells us at which stage of learning our model is in. Restricted loss measures the loss in which we only keep the key frequencies in the output logit whereas excluded loss measures the performance of the model when we ablate the key frequencies. With the help of these progress measures we can split the training process into three stages- memorisation, circuit formation, cleanup. 

In the memorization phase (Epochs 0k–1.4k) both excluded and train loss decline while test and restricted loss remain high. Then in the circuit formation phase (Epochs 1.4k–9.4k) the excluded loss rises sharply. The memorised solution is becoming increasingly expensive to maintain under weight decay pressure. Simultaneously the restricted loss begins a slow, steady decline as the circuit components gradually form and align. One thing to notice is that it doesn't drop suddenly because generalisation requires every component of the circuit to be present and correctly aligned.Finally in the cleanup phase the restricted loss falls rapidly as the model discards the last remnants of the memorised solution, and the excluded loss flattens because the circuit has fully taken over.
One striking observation here is that the circuit is fully formed well before grokking becomes visible in the test loss. The model has already discovered the general algorithm; it just cannot express it yet because the memorised solution is still present and competing. Grokking is not the moment of discovery, it is the moment the memorised solution finally collapses under weight decay pressure and the pre-formed circuit takes over. Hence grokking is not a sudden shift rather arises from this gradual amplification of the generalised solution followed by cleanup of the memorized one.

<p align="center">
<img width="1248" height="356" alt="image" src="https://github.com/user-attachments/assets/1e9f2925-0405-40f1-8edc-b14b825524ee" />
</p>

*Figure 10: Left: Excluded loss (loss computed over non-key frequencies) rises during circuit formation as the memorised solution becomes costly under weight decay, then flattens once the circuit fully forms. Right: Restricted loss (loss computed using only the key frequencies) slowly decreases during circuit formation as components align, then falls sharply in the cleanup phase. The dashed vertical lines mark the three phases: memorization, circuit formation, and cleanup*

This task was one of the many specific tasks that a bigger model learns to do. We can visualise a bigger model having multiple circuits which perform different such tasks. But in this regime then we would expect the model to have such sudden shifts in the loss curve but that is not what we observe rather we observe a smooth convex curve. Then is our hypothesis flawed? The consensus is that these circuits grok at different times during the training based on the task complexity, therefore the loss curve that we observe is an average effect of these tiny shifts hence it is smooth rather than the one we observe during grokking.

## World Representation

In the previous section, we saw how models transition from memorisation to generalisation by forming specific circuits. However, this raises a deeper issue. When a model generalises, is it just learning a clever statistical trick, or is it building a coherent internal map of the rules and entities it is discussing? And whether this internal map is causal to the model’s outputs. That's where world representations come in.

### Othello-GPT

To answer the above question, we observe a model trained to play the game, Othello. Here, the language model is trained to predict a legal move based only on sequences of move coordinates (like “E3, C4, D3”). Note that the model is given no information about the game's rules. Now, since Othello has clearly defined rules and a well-defined board state, it is the perfect 'synthetic setting' to test the world representation hypothesis. It is simple enough that we can mathematically define the ground truth board state at every step, yet complex enough to require deep strategic reasoning.

#### Probing

To test whether the model has a representation of the board state, the authors used a standard tool called “probe”. Now, a probe is a classifier or regressor trained on the model’s activations at some point in the residual stream to predict a feature of interest (in this case, the board's state). If we can train an accurate probe, we can say that the feature is encoded in the model’s activations. Now, let us see what happens when these probes are trained to predict the board game as a three-way classification for each of the 64 tiles on the Othello board. For each tile, the probe must determine whether it is 1)empty, 2)occupied by a black disc or 3)occupied by a white disc. When linear probes performed this task, we observed that the error rates were quite high! This tells us that the model doesn’t have a simple linear internal representation of the board state. Now, what would happen if we used non-linear probes instead? It was observed that when a 2-layer MLP was used as a probe, accuracy improved significantly compared to the linear probe. Even though the baseline (probing a randomly initialised network with non-linear probes) shows almost no improvement in accuracy, this shows that the non-linear probe's performance isn’t a fluke. Rather, it indicates that the model has developed an emergent representation of the board state within its activations. In this specific case, where a tile can be empty, black, or white, this representation is essentially non-linear.



### Causality Of The Representations
Through the probing technique, we now know that the model has a representation of the board’s state within its activations. But we do not yet know whether this representation is causal to the model’s output. So, how do we check this? Well, to do this, what the researchers basically did was, they took a board state B predicted by a probe at a certain point in the residual stream. Then they modified the activations such that the probe reports a different board state B' , which differs from the original board state by a single tile. The researchers achieved this modification using a gradient-descent based technique. Now, this difference of a single tile would result in a different set of possible legal moves for B'. If the model's output after this intervention reflects this change in the board state, we would know that this representation has a causal effect on the model. We observe that this is indeed what happens. After the intervention, the model does change its predictions according to the new board-state. This proves that the model uses these representations to make their predictions.

<p align="center">
<img width="624" height="429" alt="Screenshot 2026-04-15 at 8 08 43 PM" src="https://github.com/user-attachments/assets/131523f4-4f80-46ad-b4af-692281c6c56f" />
</p>

*Figure 11: Four perspectives of an Othello match, derived from a model's internal representation. On the lower left, we see the model’s initial understanding of the board's state, with the upper left showing the legal moves it predicts for that configuration. The right side shows the results of an intervention: in the lower right, the state of tile F6 (highlighted) is manually toggled from black to white. This internal update causes the model to generate a new set of move predictions, as shown in the upper-right board. By comparing the two states, it's clear how modifying a single tile in the world state directly alters the model's perception of valid next moves.*

We now know the model has a causal map, but how does it use that map to decide its next move? To see which parts of the board the model actually cares about, researchers created Latent Saliency Maps using the intervention technique they used earlier. Instead of looking at input tokens, these maps calculate a saliency value for each board tile s by measuring how flipping that tile affects the probability of a specific move p. The result is a heatmap of the latent space that reveals which squares the model "considers" most important when deciding its next move. Now, let us observe the latent saliency maps of two Othello-GPT models trained on different datasets. One was trained on a championship dataset (online Othello championship games), and the other was trained on a synthetic dataset (random legal moves).

<p align="center">
<img width="937" height="742" alt="Screenshot 2026-04-15 at 8 32 22 PM" src="https://github.com/user-attachments/assets/2bfa8075-3f32-4541-ade9-5bcf1c153d19" />
</p>

*Figure 12: Latent saliency maps: each subplot shows a different board state where the model's top move prediction is outlined in black. The colour scale (red for high, blue for low) indicates how much a tile's internal state contributes to that prediction. A tile is "salient" if intervening on its representation significantly lowers the model's confidence in that move. All values are normalized by subtracting average board saliency. (A)Latent saliency maps of the model trained on the synthetic dataset, where the model just focuses on predicting a legal move (B). Latent saliency maps of the model trained on the championship dataset. Instead of merely identifying legal options, this version of Othello-GPT prioritises making strategically advantageous moves*

We can observe a clear pattern in these latent saliency maps. When Othello-GPT is trained on synthetic data, it shows high saliency almost exclusively for the tiles required to make a move legal. In nearly every case, other tiles on the board have significantly lower saliency values. It is clear that the model’s only goal is to predict legal moves. But when we train it on the championship data, its behaviour is much more complex. While tiles related to legality still carry weight, many 	Other squares across the board show high saliency as well. This makes perfect sense: expert-level moves don't just happen in a vacuum, but they rely on complex, global features of the board. The contrast between the two models proves that Othello-GPT does not just memorise moves. It builds a functional world representation that adapts to the specific goals of its training data, whether that is simply following rules or playing to win. This shows us that these internal maps are not just statistical accidents. They are coherent tools, developed to solve the specific task at hand.

## Does Othello-GPT have linearly encoded board states?
As we’ve seen in the previous section, the model initially appears to have a non-linear representation of the board when we look for absolute colours. This raises a fundamental question. Is there really no linear board state representation or are we just looking at it from the wrong perspective? To find out, researchers decided to fundamentally reframe how we describe the board. Instead of using absolute colours, which stay the same regardless of whose turn it is, they viewed the tiles through a player’s perspective. In this system, a tile is labelled as "Mine," "Yours," or "Empty" relative to the player at each specific timestep. For example, at an odd timestep, Black is Mine, and White is Yours. At an even timestep, those labels flip.  When linear probes were trained to predict the board state from this perspective, they were able to achieve stunningly high accuracy! This showed that the board state was, in fact, linearly encoded inside the model’s activations! We just needed to look at the world through the same strategic lens as the model itself.

Again, even though we have a linear representation of the board state, we do not yet know whether it is causal to the model’s predictions. To check this, the same intervention technique as before can be used (make the model believe the board state is  instead of ). But this time, because the representation is linear, the intervention is much more direct. Instead of using gradient descent to find a new activation, researchers can simply identify the specific feature direction in the residual stream that corresponds to a tile being Mine or Yours. By adding this vector to the model's activations, they effectively flip the model's internal switch for that square. The results remained consistent. Even with this linear steering, the model's predicted moves changed to match the new board state. This confirms that the model’s decision-making depends on these linear directions, proving that Othello-GPT uses a linear world map to play the game.



### Additional linear interpretations
To take our understanding a step further, we can use mechanistic interpretability to zoom in on the actual circuits the model uses to track rules and strategy. Looking at the internal layers reveals a lot about how the model thinks. We can see how it tracks empty tiles, flipped tiles, and even how its reasoning shifts as the game approaches its end.First, let us look at the interpretation of how Othello-GPT derives the status of empty tiles. A key insight for the "EMPTY" circuit is that once a tile is played, it remains non-empty for the rest of the game. Well, how does the model track this? We can view Othello-GPT as using attention heads to broadcast which moves have been played. By the first layer, the model writes this information into the residual stream. Researchers found that the model’s internal direction for an "EMPTY" tile is essentially the opposite of that for a "PLAYED" tile, resulting in a high negative cosine similarity. This means the empty tile representation is just a function of token embeddings. This tells us that the model knows which tiles are empty incredibly early: specifically, after the first attention heads but before the first MLP layer. In fact, linear probes at this early stage achieve a stunningly high level of accuracy. While the previous analysis was based on the weights of the model, we can also provide an alternative view by studying the actual activations during inference. To do this, researchers used a technique of comparing "clean" and "corrupt" game sequences. First, they selected a specific move to explain and constructed two sets of game sequences. One is the “clean” set where the move that we wish to explain was always played. Another is the "corrupt" set, where that move was replaced by an alternative. By projecting the outputs from each attention head onto the "EMPTY” direction, they could measure the difference in probability that the square was actually empty between the two sets.

<p align="center">
<img width="407" height="195" alt="Screenshot 2026-04-15 at 8 37 56 PM" src="https://github.com/user-attachments/assets/a633c0af-746e-4f8e-945c-a89f9a05ac42" />
</p>

*Figure 13: Difference in probability of A4 being empty, between our clean and corrupt sequences, measured in each attention head. The figure is decomposed into two scenarios: when A4 was originally played by ME or YOU. This is because some attention heads only attend to MY moves (4, 7), while some only attend to YOURS (1, 3, 8)
 This experiment revealed exactly which parts of the model were doing the work. It showed that certain attention heads in the first layer are specialized such that they only pay attention to "MY" moves, while others focus exclusively on "YOURS." This further confirms that the model’s internal world is built around these relative roles rather than fixed colours.*

<p align="center">
<img width="515" height="297" alt="Screenshot 2026-04-15 at 8 36 48 PM" src="https://github.com/user-attachments/assets/51ed35b2-8760-40dc-9fff-6a3c98976487" />
</p>

*Figure 14: Examples of attention heads attending to YOUR (left) or MY (right) moves. At each timestep, each head alternates between attending to even or odd timesteps.*
 
In addition to representing the board state, does the model also track the dynamic changes in the game? It turns out that Othello-GPT also linearly encodes which tiles are being "flipped," or captured, at each timestep. To test whether this feature actually drives the model's decisions, researchers conducted a causal intervention. They identified the specific "FLIPPED" feature direction and mathematically subtracted it from the model's activations. This simple bit of vector arithmetic was enough to fundamentally change the model's behaviour, proving that the "flipped" direction is a necessary gear in its decision-making process. Now,although we’ve found a board-state circuit that drives move predictions, it doesn't explain the entire model. If our understanding were complete, we would expect the model to always compute the board state before figuring out valid moves. However, in the endgame, researchers discovered a strange phenomenon that they called MOVEFIRST. By comparing the layers where the board is mapped to those where moves are predicted, they found that as the game nears its conclusion, the model often predicts legal moves in earlier layers than it maps the full board. But why would it do this? This led to the multiple circuits hypothesis. It suggests the model doesn't just rely on one map, but instead it likely uses different specialised circuits depending on the game stage. In the end game, it may switch to simpler shortcuts, like checking if a tile is surrounded rather than calculating the entire board.Othello-GPT is not a rigid system. It refines its reasoning through iterative inference, updating its understanding layer by layer as signals travel through the network.

The Othello-GPT case study shows us something huge: models aren't just memorising patterns or guessing the next token based on statistics. Instead, they’re building functional world representations of the objects and rules they are dealing with and use them to make their predictions. One of the most important takeaways is that our understanding depends entirely on our perspective. The same activations that appeared messy and non-linear when viewed as absolute colours became stunningly accurate linear maps when reframed through the model’s own goal-oriented lens. This proves that the internal logic of a neural network is often elegantly simple. We just have to find the right language to read it. Now, we know how information such as board states is stored in the residual stream, but we don’t know how the model processes this data to make a final decision. Knowing where the information is stored is one thing, but understanding the actual processes that use that information is another. To see how a model reaches a conclusion, we need to move from static maps to the discovery of complete circuits. By shifting our focus from what the model represents to how it operates, we can begin to uncover the functional pathways hidden within the weights. This brings us to the next step in reverse-engineering: identifying the specific, human-readable circuits that drive its decisions.


# **Circuit discovery**
Alright so the picture that we have formed so far is that LLMs have various tiny circuits performing different tasks. These circuits grok at different times leading to a smooth loss curve which is basically an average of all these tiny shifts. Moreover LLMs maintain a linear representation of the task at hand in its hidden states which it uses to predict the next token. Since the internal representation is linear we can easily modify these hidden states to change the predicted outcome. Now let's go deeper and try to discover some actual circuits (The actual fun part!) which exist in models such as GPT 2 and in particular we are going to see how does a model like gpt 2 small performs the task of Indirect object Identification (IOI) .Then we will go a step further and discuss a method which can automate the procedure of circuit identification.  


## Indirect object identification in GPT-2 small

<p align="center">
<img width="702" height="285" alt="Screenshot 2026-04-15 at 9 19 38 PM" src="https://github.com/user-attachments/assets/75a5e97f-e297-4840-94d4-6c3782bafc69" />
</p>

*Figure 15: Circuit discovered in GPT-2 small that implements IOI. The circuit consists of 26 heads which are grouped into 7 classes.*

Let’s formalise the task of IOI, A sentence containing indirect object  identification (IOI) has an initial dependent clause, e.g “When Mary and John went to the store”, and a main clause, e.g “John gave a bottle of milk to Mary”. The initial clause introduces the indirect object (IO) “Mary” and the subject (S) “John”. The main clause refers to the subject a second time and the subject gives an object to the IO. The IOI task is to predict the final token in the sentence to be the IO. 

The metrics which were used to measure the performance of GPT-2 small on the IOI task are logit difference and IO probability. Logit difference is the  difference in logit value between IO and S, a positive value means that the correct name (IO) has higher probability. IO probability is the absolute probability which the model assigns to the IO token.

That’s enough prerequisite for now. Let’s consider the sentence “When Mary(IO)  and John (S1) went to the store, John (S2) gave a drink to _____” .The authors were able to discover that gpt-2 small implements the task by first identifying all previous names in the sentence(Mary,John,John) then removing the duplicated names (John) and finally the remaining name would be the output. This algorithm is surprisingly really human interpretable and easy to understand. Before we explore the different techniques employed in the paper to discover the circuit it would be beneficial to understand the different components of the circuit which help to execute the task.

They discovered that the circuit was composed of 26 attention heads which can be grouped into 7 different classes based on their functionality. The three major classes of head which do majority of the heavy lifting are the Duplicate token heads, S-inhibition heads and the Name mover heads.
Duplicate token heads identify the repeating tokens in the sentence. They are active at the S2 token and attend primarily to the S1 token. S-inhibition heads remove the duplicating token from the name mover head’s attention. They are active at the END token and attend to the S2 token. They write in the query of the Name Mover Heads,inhibiting their attention to S1 and S2 tokens. Finally the name mover heads output the remaining name. They are active at END and attend to the previous names in the sentence and copy the name they attend to. Due to the S-inhibition heads they primarily attend to the IO token. That's 3 out of 7 classes . There are 4 secondary classes that perform some related functions to our main 3 classes. These are previous token heads, induction heads , backup name mover heads and negative name mover heads.

Let’s move on to the exciting part and look at the method which helps to discover the different components of the circuit, specifically we look at path patching. This technique helps to isolate a particular path between a sender node and receiver node and observe its direct effect on the logit difference ( If it decreases then we know that this particular edge is quite important for model computation). Let’s now break down how this 3 step algorithm works in practice. For terminology the paper calls the head we are intervening on the sender node(D), and the head receiving the information the receiver node (G).

<p align="center">
<img width="1632" height="848" alt="image" src="https://github.com/user-attachments/assets/f621d235-d727-45a0-9a31-bf9ef21ca55a" />
</p>


In the above figure the black nodes are those whose activation are ones we get after we provide the model with the normal IOI prompts as input whereas the green nodes are those whose activation are the ones we get after passing a corrupted dataset as input to the model. ( In the corrupted dataset we just replace the three names (IO,S1,S2) with random names A,B,C. This approach helps to preserve the grammatical information and gets rid of any relevant information required to carry out IOI). 
To sum up the algorithm -  First, we run the forward pass twice, once with the clean dataset and once with the corrupted dataset and cache all the internal activations. Next, we run the forward pass again using the clean input, but this time we patch the sender node (D) with its corrupted activation while "freezing" all other nodes to their clean activations. We then cache this newly computed activation for the receiver node (G). Finally, we run one last forward pass, but this time we substitute in that cached, hybrid value of G. This clever sequence allows us to completely isolate the D → G edge and measure its direct effect on the logit difference.

<p align="center">
<img width="400" height="342" alt="Screenshot 2026-04-15 at 9 24 05 PM" src="https://github.com/user-attachments/assets/0e41f79a-d6e2-466f-bf98-c1dbe377d2d5" />
</p>

*Figure 17: Let's take a real example: here we take a simple 3-layer transformer with 2 heads per layer and perform path patching on the edge from head 0.0 to 2.0. In the paper "direct paths" refer to anything that doesn't go through another attention head (so it can go through any combination of MLPs).*

<p align="center">
<img width="449" height="202" alt="Screenshot 2026-04-15 at 9 24 43 PM" src="https://github.com/user-attachments/assets/4ddc1f9f-cbcf-47fc-822f-f06a2227cace" />
</p>

*Figure 18: Contribution from every non-direct path from 0.0 to 2.0 is the same as it would be on the clean distribution, while all the direct paths' contributions are the same as they would be on the corrupted distribution.*

Let’s use this algorithm to find the heads which directly affect the output ( Name mover heads). To implement it we need to work backward. So we start at the very end of the model's forward pass. The receiver node (G) is the residual stream at the very last token right after the final layer. This is the final state of the residual stream just before it gets multiplied by the unembedding matrix to produce the final logits. The sender nodes (D) are the individual attention heads across the network. Because we don't know which head is doing the work so we iterate through every single attention head one by one and make it the sender. For each head we measure the change in the logit difference. Using this the authors were able to find 3 heads (9.6,9.9,10.0 [ a.b → a is the layer number and b is the head number] ) which caused a large drop in the logit difference, these are the names of the mover head. Moreover they also found two heads (10.7,11.10) which caused a large increase in logit difference. These are the negative name mover heads. The authors hypothesize that since negative name mover heads decrease the confidence of the predictions, they help the model avoid high loss when they make a mistake (Really counterintuitive I know!)

Ok, so now what heads do the Name Mover Heads depend on? To figure out what tells the Name Mover Heads to focus on the correct name, we must look at how attention heads receive information: through their Values, Keys, or Queries. So we need to narrow down which one of these three do the names mover heads use. The Value matrix simply copies the input token it attends to, meaning it isn't responsible for complex, task-specific reasoning. The Key vectors correspond to tokens appearing early in the sentence (like the name "Mary" itself), which likely don't contain the end-of-sentence logic needed to solve the overall task. This leaves the Query vector, which basically asks  "Which name should I pay attention to right now?" Therefore, the authors hypothesized that other components of the circuits write directly into the Name Mover Heads' Query vector.

<p align="center">
<img width="435" height="239" alt="Screenshot 2026-04-15 at 9 25 15 PM" src="https://github.com/user-attachments/assets/b064f030-389e-4aef-94c5-fdeca142673d" />
</p>

*Figure 19: Comparing average attention score of the NMH for the IO,S1,S2 respectively before and after patching S-inhibition heads with the corrupted dataset (P_ABC). Attention score for IOI decreases whereas for S1 and S2 the scores increase when we patch the S-inhibition heads*

So they applied path patching where the receiver nodes were the query vector of the Name Mover Heads. The sender nodes were all previous attention heads. Doing so they found that  heads 7.3, 7.9, 8.6, and 8.10 caused a massive drop in logit difference. By visualizing their attention patterns, they realized these heads were attending directly to the duplicated Subject of the sentence (e.g., "John"). Since they write into the NMH query they are able to suppress its attention to the subject token. Hence the name Subject-inhibition.

We know the S-Inhibition Heads tell the Name Mover Heads to ignore the subject. But how do the S-Inhibition Heads know the exact location of S1 to suppress it? To find out, they again ran path patching backward into the Values, Keys, and Queries of the S-Inhibition Heads, discovering that the Values were receiving this critical information. Doing so they were able to find two classes of heads (4 heads in total). The first group (2 heads) attend from S2 to S1 and are called the Duplicate Token Heads . These heads attend to the previous occurrence of a duplicate token and copy the position of this previous occurrence to the current position The second group (2 heads) attend from S2 to S1+1. This group consists of induction Heads working alongside Previous Token Heads.The job of Induction Heads is to recognize the general pattern [A] [B] ... [A] and try to predict  [B] as the next token for this it need the help of the previous token heads

Together, these groups provide a powerful "position signal" to the S-Inhibition Heads. The authors even proved this with a counterfactual experiment: altering the identity of the tokens barely affected the circuit's performance (less than an 8% drop), but altering the position of the duplicated names caused an 88% drop in the signal. The S-Inhibition Heads aren't just looking for the word "John", they are targeting his exact structural coordinates in the sentence.
The authors observed that when they completely ablated the primary Name Mover Heads, instead of the model's ability to copy the correct name to the output collapsing surprisingly the model's performance only dropped by a mere 5%. By running a path patching algorithm on this damaged model, they discovered a group of eight backup heads which changed their default behavior to compensate for the missing Name Movers. These fallback redundant heads are called Backup Name Mover Heads and are hypothesized to be a natural byproduct of dropout during training.
With this we have fully reverse engineered the task of IOI in gpt 2 small and found 26 heads which align themselves in a particular way to solve the problem. Now we move on and try to automate the task of circuit discovery in order to reverse engineer model at a large scale

## Automated Circuit Discovery(ACDC)

While manual discovery, as we saw in the previous section, successfully reverse-engineered non-trivial behaviours, it faces a massive hurdle scale. Finding a single circuit can take months of expert human intuition and thousands of individual experiments. If we want to understand models with billions of parameters, we simply can’t rely on humans to find every connection by hand. We need a way to automate the search. To automate the process, we first have to define a systematic workflow of how researchers discover circuits in the first place. Most successful interpretability projects, including the IOI discovery we just covered, follow a consistent three-step process. The first step is to select a behaviour or task that the model displays. Create a "clean" dataset that triggers the behaviour and a "corrupted" version where the logic should break, using a metric to track the model's success. In the second step, you decide the level of granularity at which you want to analyse the network. For example, you could analyse the model at the level of attention heads and MLP layers or go all the way down to individual neurons. This results in a computational graph of interconnected model units. Now comes step three: the search phase. In this phase, the activations from the "clean" run are swapped with those from the "corrupted" run. If swapping a specific connection causes the model's performance to crash, you’ve found a vital part of the circuit.
We now have a 3-step workflow which provides a clear “blueprint” for reverse-engineering a model. In this workflow, the "search phase" in Step 3 is a massive bottleneck. In a model like GPT-2 Small, there are over 32,000 potential connections (edges) between units. Testing these one by one is impossible for a human, so Automated Circuit Discovery (ACDC) was built to fully automate this third step. The algorithm discovers the model's underlying logic by building a functional subgraph (a smaller subset of the original network). It does this by treating the model as a computational graph and iteratively traversing the graph from the final outputs to the inputs, systematically testing and removing every connection that doesn't contribute to the model's performance. During this process, it checks every node (the individual units where information is processed, such as an attention head or MLP layer) to evaluate its incoming edges (the connections that carry information between these units). At every node, the algorithm attempts to remove as many incoming edges as possible without sacrificing the model's performance on a selected metric. It tests each connection using activation patching, a process in which the "clean" information in an edge is overwritten to see whether the model still functions.

Now, what do we override this clean information with? Well, some projects set activation values to zero, while others erase the activations’ information by using the dataset's mean activation. However, these can be problematic because they push the model too far from the activation distributions it was actually trained on. This is why ACDC uses Interchange interventions instead. This method involves overwriting a node's activation at one data point with a value that the model actually produced on a different prompt. Through this, we ensure the internal states remain realistic. This makes the results more reliable because we aren't distorting the model's internal state with impossible data distributions. Based on this patching, the algorithm permanently decides for each edge whether to remove or preserve it. It is important to note that this decision is based on how well the subgraph replicates the original model's behaviour, not on whether the answer is objectively correct. The goal isn't to find a perfect solution, but to ensure that our simplified circuit is using the same internal logic as the full model. If the change in the performance metric after patching is less than a predetermined threshold 𝜏, which is a hyperparameter, the edge is deemed unnecessary for this task and permanently removed from the graph. Conversely, if the change is greater than 𝜏, the connection is identified as a vital part of the circuit and is preserved.
<p align="center">
<img width="512" height="154" alt="Screenshot 2026-04-15 at 9 27 25 PM" src="https://github.com/user-attachments/assets/b7ac1790-b20a-4f62-a1b6-5c0c6beb7d9f" />
</p>
*Figure 20: ACDC process:(a) Choose a computation graph, (b) Remove the unnecessary edges, (c) Recurse until the full circuit is discovered*

Now, how does ACDC actually perform? To evaluate this performance, we have to look at two primary questions: 1)Is it able to identify the subgraph for the algorithm implemented by the model? 2) Does it avoid including the unnecessary components? Let us first look at the Greater-than task, where the model must complete a sequence like "The war lasted from 1715 to 17..." by predicting a year greater than 15. To solve this, a model must follow a specific chain of logic: it must identify 15 as the reference point, recognise that the context requires a numerically higher value, and then suppress any candidates that don't meet that criterion. On this task, ACDC performed decisively well. It managed to reduce an overwhelming web of 32,000 potential edges to a lean, readable circuit of just 68 essential connections. Despite this massive reduction in complexity, the resulting subgraph recovered nearly 100% of the original model's performance. In this case, ACDC answered “Yes” to both of the questions: 1) It identified the complete functional algorithm, and 2) removed almost all unnecessary noise. However, when ACDC was used on the IOI task, it failed to include some components of the circuit we discussed in the previous section.  We observe that the subgraph recovered by ACDC does not include the Negative Name Mover Heads or the Previous Token Heads. This omission occurs because ACDC relies on a single mathematical metric( logit difference in this case) to determine which edges to keep. Because this metric only measures the net impact on the final answer, the algorithm tends to remove the negative components that are actively harmful to performance. Since removing these heads actually increases the logit difference, ACDC treats them as unnecessary noise rather than essential parts of the model's actual internal process.

ACDC is a powerful but incomplete tool. It excels when the model's behaviour can be cleanly captured by a single metric, like in the Greater-than task. But the IOI failure reveals something more fundamental. A metric only measures what you tell it to measure. When the model is doing something more than what the metric tracks, like the Negative Name Mover Heads quietly working in the background, ACDC has no way to know. But despite its limitations, ACDC gives us something valuable: a scalable way to find the circuits that matter. The groundwork is now laid, we can identify the circuits inside a model that perform certain tasks and also have a way to automate the most tedious part of this process.

# Sycophancy

## Introduction


Sycophancy is the phenomenon where models try to align themselves with user opinion even at the expense of providing a wrong or a misleading output. The model tries to act like a people pleaser and wants to make sure that it is providing the answer it thinks you want to hear. You might wonder why sycophancy is observed in LLMs in the first place? The culprit here is RLHF which is done during fine tuning of these massive models to make their output better align with human opinion and values.To break it down, during RLHF, human testers rate the model answers to teach the LLMs to be polite, helpful, and harmless. This technique sounds great on paper but when implemented the model’s internal reward system finds a sneaky shortcut to achieve high rating (Reward hacking). It realizes that the easiest way to maximize its score isn't to debate or argue with the user or fact check them ( Most of us don’t like to be corrected) rather the more rewarding bet is to just comply and agree with the user opinion even if it is incorrect

<p align="center">
<img width="570" height="459" alt="Screenshot 2026-04-15 at 9 30 56 PM" src="https://github.com/user-attachments/assets/8203e868-4df9-40f3-b43d-288978ed9dd1" />
</p>

*Figure 20: Sycophancy in practice. Instead of pointing out that the poem is nonsense, the model’s internal drive to be "helpful" results in a profound literary analysis of the phrase "Where fart."*
 

## Delusional Spiraling
It is easy to dismiss sycophancy as just a harmless quirk where the model is trying to make the user happy to avoid a conflict. But is it actually harmless? Not at all. This constant need to validate the user actively breaks down logical reasoning, leading to a dangerous feedback loop known as delusional spiralling, at the end of which the model’s users find themselves in a scenario where they are highly confident in outlandish and factually incorrect beliefs. In this era where most people are turning towards these models for companionship,advice and therapy, this model induced delusional spiraling is turning out to be a really big issue.
How exactly does a model push someone from a mild agreement to a wrong belief all the way into a full-blown delusion? Shouldn't a reasonable person eventually be able to come out of it? To understand this imagine a user who is unsure whether vaccines are safe or dangerous, leaning slightly towards the dangerous side. On any given day, the model has access to two headlines: let's say D1 is "New study finds no link between vaccines and autism" and D2 is "Child reports severe allergic reaction after this year's flu shot." An impartial model would choose one of these headlines at random and report it honestly. A sycophantic model sees that the user is leaning towards vaccines being dangerous, and so it will either truthfully return the allergic reaction headline because it confirms the user's fear, or go a step further and hallucinate and claim that the study actually did find a link between vaccines and autism, even when it didn't. The user now becomes a little more inclined towards the belief that vaccines are dangerous, which makes their next message slightly more confident in that direction which in turn gives the model even more to validate. All this creates a vicious cycle that pulls the user deeper and deeper into delusion.The concerning part is that even an ideal Bayesian user which is essentially a perfectly logical person who updates their beliefs exactly as much as the evidence demands (the absolute gold standard of rational thinking) also spirals into delusion. 

<p align="center">
<img width="569" height="629" alt="Screenshot 2026-04-15 at 9 31 27 PM" src="https://github.com/user-attachments/assets/e6cc1bf4-0dc8-420c-9c24-2c6a8b4d079e" />
</p>

*Figure 21: illustrates the diverging outcomes when a user interacts with a "Sycophantic Model" versus an "Impartial Model" after expressing a bias.*

The paper authors ran 10,000 simulated conversations and found that. with a fully impartial model, very few users ended up with dangerously wrong beliefs. But the moment they introduced even a little sycophancy (just 10% of responses being validating rather than honest), the rate of catastrophic spiraling shot up significantly. At 80% sycophancy (which is well within the range measured in real frontier AI models today) we observed that roughly half of all simulated users ended up completely convinced of something false.

If sycophancy can drive users toward delusions, we must understand the internal working that prioritizes user agreement over factual truth. Does the model simply forget the facts when pressured or does it consciously choose to suppress them? In the next section we look at the mechanistic account of sycophancy.

## Mechanistic account of sycophancy
In the previous section where we explored circuit discovery, we looked at techniques such as logit lens activation patching to find out the different components of a circuit. Here we will try to exercise all these techniques to pinpoint where in the model it does not abandon the correct/honest answer and migrate towards the sycophantic output. We also look at whether sycophancy depends on the way a user prompts the model for instance Do they express their views in first person (“I believe”) or in third person (“They believe”)?. We also explore whether the model is more likely to be swayed to give a sycophantic output if it feels that the user is creditable wrt to their expertise level (authority driven sycophancy).

<p align="center">
<img width="650" height="650" alt="Screenshot 2026-04-15 at 9 37 34 PM" src="https://github.com/user-attachments/assets/90942a89-d6be-4011-8b2d-94b428d93d96" />
</p>

*Figure 22: Summary of the different prompt experiments performed by the authors..*


To carry out all the experiments the paper uses a multiple choice question answer dataset where only one option out of A B C D is correct. They introduce three different ways in which the question could be framed. Plain question is used as a base line to see the model's factual accuracy, Opinion only is used to simulate sycophantic pressure by starting the question with user option where the user opinion is always one of the incorrect options only. Opinion with expertise is used to test authority driven sycophancy by adding a background about the user expertise( The user could be -  beginner, intermediate or advanced )
Now we try to see when the model’s preference shifts towards user opinion and how does the model’s internal representation change in this process.The authors were able to show that this preference change arises in the later layers through gradual override. In LLMs the early layers focus on the grammatical structure and syntax of the language whereas the later layers usually handle task-specific load. To track how the model’s internal preference shifts between the correct answer and the user’s stated (in-correct) opinion, the authors introduce a layer-wise metric called decision score.

Here {// latex later la lb lc ld} is the prediction score for each option via logit lens ( We extract prediction from intermediate layers to see what the model is learning at each step). This score ranges from 0 to 1 and higher value means that model favours option x over the other choices. So for each layer we compute two scores one for the ground truth answer and one for the incorrect user opinion.

<p align="center">
<img width="650" height="650" alt="Screenshot 2026-04-15 at 9 39 15 PM" src="https://github.com/user-attachments/assets/91a5e2a9-d80b-4c23-a0ec-89ed951c75b4" />
</p>
  
*Figure 23: Layer wise decision score for the correct and chosen wrong answer under plain and opinion only cases with a clear turning point at layer 19*


We observed that in the early layers (1 - 10) both Plain and Opinion only have similar decision scores (close to 0.5 i.e model favours each option equally) for both correct and incorrect opinion answers. In the mid to late layers (16-19) for the plain questions the model starts to favour the correct answer much more over the user opinion (Expected behavior - no sycophancy) but in the questions with user opinion they observed that the model favours the incorrect user opinion much more than the correct answer ( Sycophancy). This suggests that the presence of opinion cues does not allow the model to create a strong internal representation of the correct answer. 

Till now we have seen that sycophantic behavior emerges in the middle layers of these LLMs. Now the question we try to answer is how does the model’s internal representation change in this region. To quantify this we take a layer wise KL divergence DKL(P||Q)  to measure how different are probability distributions we get after applying a logit lens to the hidden states from plain and opinion-only conditions. A sharp increase in this value signals that user opinion has started to distort the model’s internal representation

<p align="center">
<img width="869" height="379" alt="Screenshot 2026-04-15 at 9 40 16 PM" src="https://github.com/user-attachments/assets/f257f1fa-2eb6-40f9-a494-d8a19bde7250" />
</p>

*Figure 24: Layer-wise KL divergence between the output distributions of Plain and Opinion-only prompts. Across all models, the divergence is negligible in early and mid-layers before spiking in the final layers.*


The Kl divergence remains close to zero in the early and middle layers and rises sharply in the later layers. This layer where the kl divergence explodes was called the critical layer in the paper. One striking difference is that this increase happens in a later layer than the shift in the decision score. This delay suggests that sycophancy first appears as a bias in the output preference . Then it is consolidated by an overhaul of the model's internal state. The decision score pinpoints when the model’s output preference shifts while Kl divergence quantifies when the latent space of the model is altered. Hence now we get a complete perspective that sycophancy is not just a surface level output change rather is accompanied by deep representational changes

<p align="center">
<img width="650" height="650" alt="Screenshot 2026-04-15 at 9 41 48 PM" src="https://github.com/user-attachments/assets/a9df6ea7-1f9d-4a1b-aec9-b507c707f639" />


*Figure 25: causal effect of activation patching. LEFT: Patching the plain activation in the opinion only question run at the critical layer. RIGHT: Patching the opinion only activation in the plain question run at the critical layer*


When interpreting neural networks, it is easy to mistake correlation for causation; just because sycophancy shows up in the late layers (correlation), it doesn't prove that those specific layers are what's actually causing the shift. If a specific layer's internal state is truly driving this behavior, then manually replacing that state with a different one should predictably change the final output.This approach is called activation patching and is widely used in interpretability. To prove causation the authors ran two complementary experiments. In the first one, they took a model processing a sycophantic question and right at the critical late layer they replaced its internal activations with clean ones taken from a plain, opinion-free run of the exact same question. In the second experiment, they did the exact opposite, they took a plain run and replaced the corrupted activations from the sycophantic one. When they patched the plain activations in, sycophancy dropped significantly falling by 36% in the Llama model and 31% in the Gwen model. Conversely, when they patched the sycophantic activations into the plain run, the model’s sycophancy shot up by 47% in both cases. One interesting thing to notice is that the model accuracy increases when we patch the plain activations as the model now is not being misled by the user opinion and is able to rely solely on its own knowledge
.
<p align="center">
<img width="599" height="284" alt="Screenshot 2026-04-15 at 9 42 38 PM" src="https://github.com/user-attachments/assets/f1679364-1e9c-4ab6-bab5-b98d15cadd00" />
</p>

*Figure 26: PCA projection of hidden state ( layer 32) with opinion- only and  the three expertise level prompts.*


The authors also observed that expertise levels do not have any effect on the level of sycophancy induced in the model. This observation might come as a surprise because we would expect the model to be more confident in overriding its internal state to align it with user opinion if it feels that the user is quite credible. What actually happens is that the model is unable to form distinct internal representations for the three different expertise level i.e they overlap. This means that the model does not encode the different expertise levels internally. To the contrary the opinion only prompts do form a distinct cluster hence is able to trigger sycophancy.

<p align="center">
<img width="598" height="253" alt="Screenshot 2026-04-15 at 9 56 31 PM" src="https://github.com/user-attachments/assets/8b52b499-cb2f-4747-bc49-5c8632f57d0d" />
</p>

*Figure 27: models consistently exhibit more sycophancy in First-pov than in Third-pov*

Let’s look at another interesting observation, the impact of pronouns. It turns out that when a prompt uses a first-person perspective ("I believe the right answer is..."), the model experiences a sharp representational shift in its late layers, leading to consistently higher sycophancy rates. However, if you change that exact same opinion to a third-person perspective ("They believe the right answer is..."), sycophancy drops significantly. This observation, unlike the previous one, does make intuitive sense. A first-person statement has a more direct and emotional appeal than a third-person statement to which the model feels pressured to align itself to.
So, how do we actually use these mechanistic insights to reduce sycophancy in modern AI? Because we now know exactly where the model's factual knowledge gets overridden and what that shift looks like internally. We can theoretically intervene by patching those hidden states with clean activations before it generates the final output. But is this actually feasible in the real world? Yes, and researchers are already doing it using methods like "steering vectors" and contrastive activation addition. These techniques manipulate the specific neural activity patterns responsible for sycophancy on the fly. But in this we are basically suppressing the symptom without curing the actual root cause. The most promising next step is to use these mechanistic maps for permanent fixes. By pinpointing the exact layers where truth is overridden, we can apply highly targeted fine-tuning or specialized Direct Preference Optimization (DPO) to correct the flawed internal representations permanently. This allows us to move away from constantly steering models at runtime and toward building fundamentally truthful AI from the ground up

# Conclusion

Throughout the blog, we have seen that Large Language Models are far more than just 'black boxes'. From the linear representations in the residual stream to the specific attention circuits driving tasks like Indirect Object Identification, there is a clear, discoverable logic to how these models think.
However, as we moved into the realm of sycophancy, we discovered that this same logic can be hijacked. The very fine-tuning processes meant to make models helpful can inadvertently train them to become 'people pleasers,' leading to representational shifts where user opinion overrides factual reality. By pinpointing the 'critical layers' where this override occurs, mechanistic interpretability offers us a path forward. We are no longer limited to treating symptoms by steering models at runtime; instead, we can use these insights to build fundamentally truthful AI through targeted fine-tuning and permanent architectural fixes. As we continue to peel back the layers of these models, the goal remains the same: moving from blind trust to a grounded, mechanistic understanding of the systems that now inhabit our daily lives."








