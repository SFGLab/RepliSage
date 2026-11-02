# The story of RepliSage and the role of AI

## How it began

RepliSage was born inside an **OPUS grant on replication stress** (Polish National Science Centre, 2020/37/B/NZ2/03757), under the supervision of **Karolina Buka** and **Dariusz Plewczyński**. The question we started with is easy to say and hard to answer: *what happens to the three-dimensional folding of a chromosome when the copying of its DNA is disturbed?*

At first there was no model at all, only curiosity. I wanted to start from the **first principles of DNA replication**, and I wandered through many ideas: the synchronisation of replication origins, chimera states, and other ways to describe many coupled units that fire in a pattern. They were beautiful to think about, but each of them rested on assumptions that nobody could check.

## The path that worked

The most direct road was, in the end, the one next to my own earlier work. I already had **LoopSage**, a Monte Carlo model of loop extrusion by cohesin. If loops give chromatin its structure, and replication forks travel along the very same DNA, then the natural step was to let the forks walk *into* the loop model. Extending LoopSage with replication dynamics gave us **RepliSage**: loop extrusion, a replication programme and epigenetic states, all evolving together under one energy function.

## It was harder than it looked

I had expected this to be easy. It was not, mostly because the knowledge we needed did not exist yet. How do epigenetic states propagate behind a fork? What really happens when a loop-extruding cohesin meets a replisome? Nobody could simply tell me. Many choices in the model are therefore honest assumptions that we debated, tested, and kept or discarded.

What you see here is the result of **years of work and many long conversations** with biologists, mathematicians and physicists, who patiently explained their fields to me and questioned mine.

## Thank you

I am deeply grateful to the whole team: to Karolina Buka and Dariusz Plewczyński for their guidance and biological insight, to my co-authors, and to every colleague who shared their time and knowledge. The full list of authors and funding sources is in the paper.

## The role of AI

**Claude** helped substantially with the work *around* the model: reorganising the code, designing cleaner visualisations and reports, and polishing this documentation. I am thankful for that contribution.

The ideas, the equations and the assumptions, however, come from human thinking and human disagreement. If you suspect that RepliSage is "AI slop", I can tell you it is not. One proof is that RepliSage is a **perfectly imperfect model**: its [limitations](limitations.md) are stated openly, and each of them marks a place where we did not know the answer and chose to say so rather than hide it.

If you find something wrong or missing, please open an issue. Improving the model remains a human conversation, and you are welcome to join it.

*Sebastian Korsak*
