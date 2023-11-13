#### QEC - general
* Final measurement is not done on stabilizers but on data qubit themselves! Can still be used to infer "perfect" stabilizer measurements! I guess we do it like this to get an accurate label...?
  * Perhaps used to collapse data qubits as a means of initializing the next experiment?

#### Correlation matrix $p_{ij}$ (https://arxiv.org/abs/2102.06132)
We can compute the correlation matrix as 
$$
p_{ij} \approx \frac{\langle x_i x_j \rangle - \langle x_i \rangle \langle x_j \rangle} {(1 - 2 \langle x_i \rangle) (1 - 2 \langle x_j \rangle)},
$$
where $x_i \in \{0, 1\}$ based on having a detection event and $\langle \cdot \rangle$ denotes an average over all experiments. 

We have three uncorrelated processes:
* Node $i$ flips parity, $x_i \rightarrow 1 - x_i$, with probability $p_i$
* Node j flips parity, $x_j \rightarrow 1 - x_j$, with probability $p_j$
* Both nodes flip with probability $p_{ij}$

Could use $p_{ij}$ matrix to match simulations against experimental data from a chip. 


#### Python
* *args = tuple of inputs to function
  * Ex: `def my_sum(*args):`, here we can iterate over `*args`, a variable number of inputs.
* **kwargs = keyword/named arguments as inputs to function
  * Ex: `def concatenate(**kwargs):`, can iterate using `for arg in kwargs.values():`.
  * Input should be of form `concatenate(a="real", b=2, c="hej")`.
* Ordering matters! Need to order: `def fun(a, b, *args, **kwargs):`
* The asterisk `*` can be used to unpack arguments! 
  * Ex: `def fun(a, b, c)` can be called with `x = [1, 2, 3]` using `fun(*x)`.
  * Split list: Split `x = [1, 2, 3, 4, 5]` using `a, *b, c = x` gives `a = 1, b = [2, 3, 4], c = 5`
* The double `::step` operator steps through an iterable in `step` long steps. Follows from `x[start:stop:step]` where we just don't define `start` or `stop`!
* bitwise operator `<<`, increased number in steps! Ex: `3 << 3 = 24` since `3*2*2*2 = 24`. As `3 * 1 << 3`?

