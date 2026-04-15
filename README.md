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

<img width="344" height="164" alt="Screenshot 2026-04-15 at 7 07 06 PM" src="https://github.com/user-attachments/assets/c10afab9-711d-4c14-b8bb-644b01a2d7a9" />


While a layer's internal computations are non-linear, the Residual stream itself maintains a highly linear structure. Every layer performs a linear operation to “read” from the residual stream. After performing its own operations - which may be non-linear - inside the residual stream, it again performs a linear operation to “write” its output onto the residual stream. A powerful consequence of this linearity is the emergence of Virtual Weight.

### Virtual Weights
Since the residual is linear in structure, we can mathematically skip the stream while seeing the interaction between any two layers. For example, let  be the projection matrix through which layer 1 “writes” into the residual stream, and  Let be the projection matrix through which layer 3 “reads” from the residual stream. Then, multiplying these two weights yields “virtual weights”  connecting layer 1 and layer 3. Now, these virtual weights quantify the amount of “signal” that flows directly from layer 1 to layer 3. Because the stream is additive, this direct connection exists regardless of what is happening inside layer 2.

<img width="274" height="288" alt="Screenshot 2026-04-15 at 7 20 18 PM" src="https://github.com/user-attachments/assets/280c74bc-daf1-4b75-b2b5-bbed31d44a95" />

### Linear Representation
It has been observed that many high-level concepts are represented as linear directions inside a model's representation space. By high-level concepts, we mean, for example, if the text is about an object, is it black or white? If it mentions a person, is the person male or female? etc..Because these concepts are linear directions, we can use them to control the model’s behaviors. This is known as model steering. If we identify the vector for a specific concept, we can manually add it to the residual stream during processing. For example, adding a “black” vector can force the sentence to be about a “black sofa” rather than a “white sofa”. Similarly, adding a “female” vector can force a model’s output to be about a “queen” rather than a “king”. This allows us to edit the model's "thoughts" in real-time without retraining its weights. Steering shows that these linear directions are not just correlations but rather causal to the model's decisions.


### Superposition
We know that a space with n dimensions can only hold n perfectly orthogonal vectors. But high-dimensional space has a unique property. In a high-dimensional space, it is possible to have an exp(n) number of almost orthogonal vectors. This allows the model to contain far more concepts than there are dimensions in the residual stream. We call this phenomenon Superposition. The model "packs" high-level concepts into the stream by allowing their directions to overlap slightly. For this packing to work without excessive interference, the features must be sparse. In fact, most of the important concepts the model learns are inherently sparse. These overlaps between concepts make mechanistic interpretability difficult. Because concept directions are not aligned with individual neuron axes, the neurons become polysemantic. This means we cannot look at a single neuron to find a single concept, as that neuron’s activity is likely shared by many different, overlapping directions.


<img width="324" height="312" alt="image" src="https://github.com/user-attachments/assets/d3fe3586-3de4-4344-bbd6-d71948bd2289" />


*Figure 1: In superposition, a neuron becomes polysemantic when the number of features exceeds the number of neurons. Because it is impossible to align the feature directions to the neuron axes*	


## **Grokking**

<img width="986" height="192" alt="image" src="https://github.com/user-attachments/assets/514f003b-48a4-4369-9aa6-3b8ab8c54f2a" />

Figure 1:Characteristic grokking loss curve trained on modular addition data, showcasing a clear dip in test loss (red) from 8k epoch to 14k

How do LLMs learn something new? Do they memorize, or can they find a general solution? Grokking is the phenomenon of  sudden transition from memorization to a general algorithm (Characterized by the sharp fall in test loss fig1)
To understand and isolate this phenomenon, we observe it on a toy task of modular addition (a + b mod p) using a small one-layer transformer (Refer to the appendix for details on the transformer architecture). We observe that in the early stages the training loss decreases sharply(expected as it is an easy task) but the test loss is high and saturates indicating overfitting where the model has memorised all answers where the weights act effectively like a lookup table,but as training continues at one point the test loss drops and becomes close to train loss indicating that the model is suddenly able to perform well on unseen examples hence reached a general algorithm.
The field of mechanistic interpretability is about reverse-engineering models to human interpretable circuits. Hence, we can think of models as having circuits that perform certain tasks. Now, let's imagine our modular addition task has a circuit X that needs to be formed in our model to perform it. Now in the start of the training phase the model prefers memorisation solution as the components which make up the circuit are not yet formed and lined up properly therefore the average of L2 norm of the all weights in such a case is very high as they have to store a lookup table therefore to force the model towards the general solution (Circuit formation) there is a need of regularisation (weight decay) which forces the model towards lower weight norm and hence simpler and general solution. As the training continues, we reach a point where the circuit is completely formed and the testing loss drops dramatically.


## Exploring the generalised solution
Here we explore the generalised algorithm the model was able to learn which turns out to be mapping the inputs onto a circle and performing addition on the circle.

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/df8a293b-d521-47ac-9768-83276b229749" />

Formally the algorithm involves projecting two integers a and b into its corresponding rotation using the embedding matrix. The embedding matrix just like all the other weight matrices in the model only contains some sparse set of key frequencies wk in the fourier domain. The attention and the mlp layers learn to calculate the sine and cosine of a+b and the final unembedding and output projection matrices then calculate the output logit c. Now when we take the softmax of the logits to make our prediction and take the most probable answer what we get is basically c = a + b (mod p). ( cosine(w(a+b-c)) is maximum at a+b-c = 0)

<img width="500" height="500" alt="clock_animation-3" src="https://github.com/user-attachments/assets/ab33f6cc-0e56-446b-b3c0-a9727bf72330" />

*Figure 3 Each token (0–112) has a 128-dimensional embedding vector. To visualise all 113 of them at once, we project these vectors onto the two Fourier basis directions at the model's dominant frequency, collapsing 128 dimensions down to 2. The result is a 2D snapshot of how the model internally represents numbers, and watching it over training reveals the moment the model stops memorising and represents the numbers as different points lying on a circle.*




This solution might feel too elegant and simple.You may wonder how do we know the model is performing its computation in the frequency domain? The answer lies in examining different weight matrices. Take the embedding matrix for instance when we apply a Fourier transform along the input dimension then compute the ℓ2-norm along the other dimension and plot them we observe peaks at only few sparse frequencies(wk) this validates that the model is indeed operating in this basis. A memorisation solution would produce an unstructured embedding matrix whose Fourier distribution would spread roughly uniformly across all frequencies resulting in  a flat plot. What we observe instead is the vast majority of energy collapsed into just 3-5 frequencies. Moreover it is observed that these same frequencies are the key frequencies in other matrices as well, further strengthening the claim.

<img width="788" height="126" alt="Screenshot 2026-04-15 at 7 33 14 PM" src="https://github.com/user-attachments/assets/3db08336-e6ef-4dad-9ee7-db0321d8a044" />

*Figure 3 The embedding is sparse in the Fourier basis, and throws away all Fourier components apart from a handful of frequencies
Periodic structure occurs in all other model components such as MLP activations, attention scores for different heads as well as the residual stream.They are always periodic with one of the key frequencies that we discovered earlier*


<img width="305" height="223" alt="Screenshot 2026-04-15 at 7 33 53 PM" src="https://github.com/user-attachments/assets/3241888a-3a85-44b4-846f-8c36be08f0f0" />

*Figure 4 Neuron's activation as y sweeps from 0 to 112 with x fixed at 17. Blue is the pre-ReLU activation of a clean sine wave, showing the neuron is computing a pure Fourier feature of the form sin(ω(x+y) + φ). Red is the post-ReLU activation the negative half of the sine is clipped to zero, leaving only the positive peaks*

<img width="631" height="136" alt="Screenshot 2026-04-15 at 7 34 41 PM" src="https://github.com/user-attachments/assets/875508c8-d160-4ac7-983f-efa5833c3ec5" />

*Figure 5 Each panel shows one dimension of the 128-dim residual stream at the final token position, as y sweeps from 0 to 112 with x fixed at 17. After grokking, individual dimensions of the residual stream oscillate sinusoidally with y the model has learned to encode the input as a Fourier feature rather than as a raw number.*

Ablation study is used extensively in mechanistic interpretability to ensure that the circuit we found actually has a causal role in the model output and is not just an intermediate artifact with no real computational significance. It involves zeroing out the activations of certain components and measuring the downstream effect on model outputs. Here when we ablate the key frequencies (Figure 5), we see the loss spike up. Ablating any other frequency leaves the loss completely unchanged. This is strong evidence that the model is indeed performing the described algorithm.
Figure 5 The loss of the transformer (lower is better) when we ablate all the frequencies notice the red spikes in loss when we ablate the key frequencies 

Earlier we observed that the model has a point where it ‘groks’ and generalizes (test loss becomes close to the train loss). This process might feel abrupt but when you dig deeper it turns out to be a gradual change from the memorised to the general solution. To understand this we need some progress measures which are computed during training and tells us at which stage of learning our model is in. Restricted loss measures the loss in which we only keep the key frequencies in the output logit whereas excluded loss measures the performance of the model when we ablate the key frequencies. With the help of these progress measures we can split the training process into three stages- memorisation, circuit formation, cleanup. 

In the memorization phase (Epochs 0k–1.4k) both excluded and train loss decline while test and restricted loss remain high. Then in the circuit formation phase (Epochs 1.4k–9.4k) the excluded loss rises sharply. The memorised solution is becoming increasingly expensive to maintain under weight decay pressure. Simultaneously the restricted loss begins a slow, steady decline as the circuit components gradually form and align. One thing to notice is that it doesn't drop suddenly because generalisation requires every component of the circuit to be present and correctly aligned.Finally in the cleanup phase the restricted loss falls rapidly as the model discards the last remnants of the memorised solution, and the excluded loss flattens because the circuit has fully taken over.
One striking observation here is that the circuit is fully formed well before grokking becomes visible in the test loss. The model has already discovered the general algorithm; it just cannot express it yet because the memorised solution is still present and competing. Grokking is not the moment of discovery, it is the moment the memorised solution finally collapses under weight decay pressure and the pre-formed circuit takes over. Hence grokking is not a sudden shift rather arises from this gradual amplification of the generalised solution followed by cleanup of the memorized one.

<img width="1248" height="356" alt="image" src="https://github.com/user-attachments/assets/1e9f2925-0405-40f1-8edc-b14b825524ee" />

Figure 6 Left: Excluded loss (loss computed over non-key frequencies) rises during circuit formation as the memorised solution becomes costly under weight decay, then flattens once the circuit fully forms. Right: Restricted loss (loss computed using only the key frequencies) slowly decreases during circuit formation as components align, then falls sharply in the cleanup phase. The dashed vertical lines mark the three phases: memorization, circuit formation, and cleanup

This task was one of the many specific tasks that a bigger model learns to do. We can visualise a bigger model having multiple circuits which perform different such tasks. But in this regime then we would expect the model to have such sudden shifts in the loss curve but that is not what we observe rather we observe a smooth convex curve. Then is our hypothesis flawed? The consensus is that these circuits grok at different times during the training based on the task complexity, therefore the loss curve that we observe is an average effect of these tiny shifts hence it is smooth rather than the one we observe during grokking.

## World Representation

In the previous section, we saw how models transition from memorisation to generalisation by forming specific circuits. However, this raises a deeper issue. When a model generalises, is it just learning a clever statistical trick, or is it building a coherent internal map of the rules and entities it is discussing? And whether this internal map is causal to the model’s outputs. That's where world representations come in.

### Othello-GPT

To answer the above question, we observe a model trained to play the game, Othello. Here, the language model is trained to predict a legal move based only on sequences of move coordinates (like “E3, C4, D3”). Note that the model is given no information about the game's rules. Now, since Othello has clearly defined rules and a well-defined board state, it is the perfect 'synthetic setting' to test the world representation hypothesis. It is simple enough that we can mathematically define the ground truth board state at every step, yet complex enough to require deep strategic reasoning.

#### Probing

To test whether the model has a representation of the board state, the authors used a standard tool called “probe”. Now, a probe is a classifier or regressor trained on the model’s activations at some point in the residual stream to predict a feature of interest (in this case, the board's state). If we can train an accurate probe, we can say that the feature is encoded in the model’s activations. Now, let us see what happens when these probes are trained to predict the board game as a three-way classification for each of the 64 tiles on the Othello board. For each tile, the probe must determine whether it is 1)empty, 2)occupied by a black disc or 3)occupied by a white disc. When linear probes performed this task, we observed that the error rates were quite high! This tells us that the model doesn’t have a simple linear internal representation of the board state. Now, what would happen if we used non-linear probes instead? It was observed that when a 2-layer MLP was used as a probe, accuracy improved significantly compared to the linear probe. Even though the baseline (probing a randomly initialised network with non-linear probes) shows almost no improvement in accuracy, this shows that the non-linear probe's performance isn’t a fluke. Rather, it indicates that the model has developed an emergent representation of the board state within its activations. In this specific case, where a tile can be empty, black, or white, this representation is essentially non-linear.



### Causality Of The Representations
Through the probing technique, we now know that the model has a representation of the board’s state within its activations. But we do not yet know whether this representation is causal to the model’s output. So, how do we check this? Well, to do this, what the researchers basically did was, they took a board state B predicted by a probe at a certain point in the residual stream. Then they modified the activations such that the probe reports a different board state B' , which differs from the original board state by a single tile. The researchers achieved this modification using a gradient-descent based technique. Now, this difference of a single tile would result in a different set of possible legal moves for B'. If the model's output after this intervention reflects this change in the board state, we would know that this representation has a causal effect on the model. We observe that this is indeed what happens. After the intervention, the model does change its predictions according to the new board-state. This proves that the model uses these representations to make their predictions.

<img width="624" height="429" alt="Screenshot 2026-04-15 at 8 08 43 PM" src="https://github.com/user-attachments/assets/131523f4-4f80-46ad-b4af-692281c6c56f" />

*Fig. Four perspectives of an Othello match, derived from a model's internal representation. On the lower left, we see the model’s initial understanding of the board's state, with the upper left showing the legal moves it predicts for that configuration. The right side shows the results of an intervention: in the lower right, the state of tile F6 (highlighted) is manually toggled from black to white. This internal update causes the model to generate a new set of move predictions, as shown in the upper-right board. By comparing the two states, it's clear how modifying a single tile in the world state directly alters the model's perception of valid next moves.*

We now know the model has a causal map, but how does it use that map to decide its next move? To see which parts of the board the model actually cares about, researchers created Latent Saliency Maps using the intervention technique they used earlier. Instead of looking at input tokens, these maps calculate a saliency value for each board tile s by measuring how flipping that tile affects the probability of a specific move p. The result is a heatmap of the latent space that reveals which squares the model "considers" most important when deciding its next move. Now, let us observe the latent saliency maps of two Othello-GPT models trained on different datasets. One was trained on a championship dataset (online Othello championship games), and the other was trained on a synthetic dataset (random legal moves).

<img width="937" height="742" alt="Screenshot 2026-04-15 at 8 32 22 PM" src="https://github.com/user-attachments/assets/2bfa8075-3f32-4541-ade9-5bcf1c153d19" />


*Fig. Latent saliency maps: each subplot shows a different board state where the model's top move prediction is outlined in black. The colour scale (red for high, blue for low) indicates how much a tile's internal state contributes to that prediction. A tile is "salient" if intervening on its representation significantly lowers the model's confidence in that move. All values are normalized by subtracting average board saliency. (A)Latent saliency maps of the model trained on the synthetic dataset, where the model just focuses on predicting a legal move (B). Latent saliency maps of the model trained on the championship dataset. Instead of merely identifying legal options, this version of Othello-GPT prioritises making strategically advantageous moves*

We can observe a clear pattern in these latent saliency maps. When Othello-GPT is trained on synthetic data, it shows high saliency almost exclusively for the tiles required to make a move legal. In nearly every case, other tiles on the board have significantly lower saliency values. It is clear that the model’s only goal is to predict legal moves. But when we train it on the championship data, its behaviour is much more complex. While tiles related to legality still carry weight, many 	Other squares across the board show high saliency as well. This makes perfect sense: expert-level moves don't just happen in a vacuum, but they rely on complex, global features of the board. The contrast between the two models proves that Othello-GPT does not just memorise moves. It builds a functional world representation that adapts to the specific goals of its training data, whether that is simply following rules or playing to win. This shows us that these internal maps are not just statistical accidents. They are coherent tools, developed to solve the specific task at hand.

## Does Othello-GPT have linearly encoded board states?
As we’ve seen in the previous section, the model initially appears to have a non-linear representation of the board when we look for absolute colours. This raises a fundamental question. Is there really no linear board state representation or are we just looking at it from the wrong perspective? To find out, researchers decided to fundamentally reframe how we describe the board. Instead of using absolute colours, which stay the same regardless of whose turn it is, they viewed the tiles through a player’s perspective. In this system, a tile is labelled as "Mine," "Yours," or "Empty" relative to the player at each specific timestep. For example, at an odd timestep, Black is Mine, and White is Yours. At an even timestep, those labels flip.  When linear probes were trained to predict the board state from this perspective, they were able to achieve stunningly high accuracy! This showed that the board state was, in fact, linearly encoded inside the model’s activations! We just needed to look at the world through the same strategic lens as the model itself.

Again, even though we have a linear representation of the board state, we do not yet know whether it is causal to the model’s predictions. To check this, the same intervention technique as before can be used (make the model believe the board state is  instead of ). But this time, because the representation is linear, the intervention is much more direct. Instead of using gradient descent to find a new activation, researchers can simply identify the specific feature direction in the residual stream that corresponds to a tile being Mine or Yours. By adding this vector to the model's activations, they effectively flip the model's internal switch for that square. The results remained consistent. Even with this linear steering, the model's predicted moves changed to match the new board state. This confirms that the model’s decision-making depends on these linear directions, proving that Othello-GPT uses a linear world map to play the game.



Additional linear interpretations
To take our understanding a step further, we can use mechanistic interpretability to zoom in on the actual circuits the model uses to track rules and strategy. Looking at the internal layers reveals a lot about how the model thinks. We can see how it tracks empty tiles, flipped tiles, and even how its reasoning shifts as the game approaches its end.First, let us look at the interpretation of how Othello-GPT derives the status of empty tiles. A key insight for the "EMPTY" circuit is that once a tile is played, it remains non-empty for the rest of the game. Well, how does the model track this? We can view Othello-GPT as using attention heads to broadcast which moves have been played. By the first layer, the model writes this information into the residual stream. Researchers found that the model’s internal direction for an "EMPTY" tile is essentially the opposite of that for a "PLAYED" tile, resulting in a high negative cosine similarity. This means the empty tile representation is just a function of token embeddings. This tells us that the model knows which tiles are empty incredibly early: specifically, after the first attention heads but before the first MLP layer. In fact, linear probes at this early stage achieve a stunningly high level of accuracy. While the previous analysis was based on the weights of the model, we can also provide an alternative view by studying the actual activations during inference. To do this, researchers used a technique of comparing "clean" and "corrupt" game sequences. First, they selected a specific move to explain and constructed two sets of game sequences. One is the “clean” set where the move that we wish to explain was always played. Another is the "corrupt" set, where that move was replaced by an alternative. By projecting the outputs from each attention head onto the "EMPTY” direction, they could measure the difference in probability that the square was actually empty between the two sets.


<img width="407" height="195" alt="Screenshot 2026-04-15 at 8 37 56 PM" src="https://github.com/user-attachments/assets/a633c0af-746e-4f8e-945c-a89f9a05ac42" />


Fig. Difference in probability of A4 being empty, between our clean and corrupt sequences, measured in each attention head. The figure is decomposed into two scenarios: when A4 was originally played by ME or YOU. This is because some attention heads only attend to MY moves (4, 7), while some only attend to YOURS (1, 3, 8)
 This experiment revealed exactly which parts of the model were doing the work. It showed that certain attention heads in the first layer are specialized such that they only pay attention to "MY" moves, while others focus exclusively on "YOURS." This further confirms that the model’s internal world is built around these relative roles rather than fixed colours.

<img width="515" height="297" alt="Screenshot 2026-04-15 at 8 36 48 PM" src="https://github.com/user-attachments/assets/51ed35b2-8760-40dc-9fff-6a3c98976487" />


Fig. Examples of attention heads attending to YOUR (left) or MY (right) moves. At each timestep, each head alternates between attending to even or odd timesteps.
 
In addition to representing the board state, does the model also track the dynamic changes in the game? It turns out that Othello-GPT also linearly encodes which tiles are being "flipped," or captured, at each timestep. To test whether this feature actually drives the model's decisions, researchers conducted a causal intervention. They identified the specific "FLIPPED" feature direction and mathematically subtracted it from the model's activations. This simple bit of vector arithmetic was enough to fundamentally change the model's behaviour, proving that the "flipped" direction is a necessary gear in its decision-making process. Now,although we’ve found a board-state circuit that drives move predictions, it doesn't explain the entire model. If our understanding were complete, we would expect the model to always compute the board state before figuring out valid moves. However, in the endgame, researchers discovered a strange phenomenon that they called MOVEFIRST. By comparing the layers where the board is mapped to those where moves are predicted, they found that as the game nears its conclusion, the model often predicts legal moves in earlier layers than it maps the full board. But why would it do this? This led to the multiple circuits hypothesis. It suggests the model doesn't just rely on one map, but instead it likely uses different specialised circuits depending on the game stage. In the end game, it may switch to simpler shortcuts, like checking if a tile is surrounded rather than calculating the entire board.Othello-GPT is not a rigid system. It refines its reasoning through iterative inference, updating its understanding layer by layer as signals travel through the network.

The Othello-GPT case study shows us something huge: models aren't just memorising patterns or guessing the next token based on statistics. Instead, they’re building functional world representations of the objects and rules they are dealing with and use them to make their predictions. One of the most important takeaways is that our understanding depends entirely on our perspective. The same activations that appeared messy and non-linear when viewed as absolute colours became stunningly accurate linear maps when reframed through the model’s own goal-oriented lens. This proves that the internal logic of a neural network is often elegantly simple. We just have to find the right language to read it. Now, we know how information such as board states is stored in the residual stream, but we don’t know how the model processes this data to make a final decision. Knowing where the information is stored is one thing, but understanding the actual processes that use that information is another. To see how a model reaches a conclusion, we need to move from static maps to the discovery of complete circuits. By shifting our focus from what the model represents to how it operates, we can begin to uncover the functional pathways hidden within the weights. This brings us to the next step in reverse-engineering: identifying the specific, human-readable circuits that drive its decisions.


