## Open-source Gloriously Extensive Yaml-configuration Repository for Reimplementing Architectures of Transformers
<p align="center"><a href="https://www.youtube.com/watch?v=JacN1MzyeKo"><i>(ogey rrat)</i></a></p>

## What the hell is this?
Here is a personal repo open to the public that is dedicated to me learning all types of Transformers by looking at other people's code and research papers and pretending that I know what I'm doing when I copy them and combine them together into an unholy abomination of build-your-own-Transformer. Basically, lucidrains' [x-transformers](https://github.com/lucidrains/x-transformers) library, but taken quite a few steps further by implementing training loops and supporting different libraries such as [HuggingFace](https://huggingface.co/) and [ColossalAI](https://www.colossalai.org/)

## Why?
People say the best way of learning AI is to play with it yourself. That's what I'm doing here. Eventually, this might become a cool thing where all you have to do is specify a .yml file (either preset or custom-made) and in one command, begin to train it.

## This just sounds like a bunch of autism.
That's exactly what this is.

## To-do list:
- [x] Implement boilerplate code
- [ ] Implement YAML-based config system (use OmegaConf library?)
- [ ] Implement basic transformer architecture
- [ ] Implement what is needed to create a basic training loop
	- [ ] Feed in datasets
		- [ ] HuggingFace support
		- [ ] PyTorch DataLoader support
	- [ ] Implement tokenizer(s)
	- [ ] Implement optimizer(s)
	- [ ] Actual process for training (zero_grad, forward, loss, backward, step)
	- [ ] Save checkpoints
- [ ] Implement inferring from a model
- [ ] Implement fine-tuning script

And more which I have probably forgotten.