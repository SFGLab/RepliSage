# The story of RepliSage and the role of AI

## How it began

RepliSage was born inside an **OPUS grant on replication stress** (Polish National Science Centre, 2020/37/B/NZ2/03757), under the supervision of **Karolina Buka** and **Dariusz Plewczyński**. The question we started with is easy to say and hard to answer: *what happens to the three-dimensional folding of a chromosome when the copying of its DNA is disturbed?*

At first there was no model at all, only wonder. I wanted to begin from the **first principles of DNA replication**, and I wandered through many ideas: the synchronisation of replication origins, chimera states, and other ways to describe many coupled units that fire in a pattern. They were beautiful to think about, but each of them rested on assumptions that nobody could check.

## The path that worked

The most direct road was, in the end, the one right beside my own earlier work. I already had **LoopSage**, a Monte Carlo model of loop extrusion by cohesin. If loops give chromatin its structure, and replication forks travel along the very same DNA, then the natural step was to let the forks walk *into* the loop model. Extending LoopSage with replication dynamics gave us **RepliSage**: loop extrusion, a replication programme and epigenetic states, all evolving together under one energy function.

## It was harder than it looked

I thought it would be easy. It was not, mostly because the knowledge we needed did not yet exist. How do epigenetic states travel behind a fork? What really happens when a loop-extruding cohesin meets a replisome? Nobody could simply tell me. Many choices in the model are therefore honest assumptions, argued over, tested, and kept or let go.

What you see here grew over **years of work, doubt and long conversations** with biologists, mathematicians and physicists. They explained their worlds to me with generosity and patience, and they were never afraid to tell me when I was wrong. Every line of this model carries a little of their time.

## Thank you

From the bottom of my heart, thank you to the whole team. To Karolina Buka and Dariusz Plewczyński, for believing in this idea when it was only a sketch and for guiding it with care. To my co-authors, for the late evenings and the honest criticism. To every colleague who shared knowledge, a coffee, or a hard question. You made this work possible, and it is yours as much as it is mine. The full list of authors and funding is in the paper.

## The role of AI

**Claude** helped a great deal with the work *around* the model: reorganising the code, designing clearer visualisations and reports, and polishing this documentation. I am sincerely thankful for that help.

But the ideas, the equations and the assumptions come from human minds and human disagreement. If you suspect that RepliSage is "AI slop", I can tell you with certainty that it is not. It is a **perfectly imperfect model**. Its [limitations](limitations.md) are stated openly, and each one marks a place where we did not know the answer and chose to say so rather than hide it. That honesty is the mark of something made by people who care.

If you find something wrong or missing, please tell us. Science is a conversation, and you are warmly invited to join it.

With gratitude and love for this work,

*Sebastian Korsak*
