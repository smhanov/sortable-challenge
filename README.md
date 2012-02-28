# Sortable Coding Challenge Contest Entry

Here is my entry for the [Sortable coding challenge]().

To run it, type:

<pre>
./sortable.py
</pre>

This will read the listings.txt and products.txt file in the current folder,
and write results.txt, unclassified.txt, and outliers.txt

You should see output similar to the following:

<pre>
Reading listings.txt
Reading products.txt
Training model
Classifying product listings
K=182 Classified 9785/20100 of 20196 listings
</pre>

## Strategy
In this challenge, we are given a list of products, each of which has a name,
model number, announce date, and possibly a product family. The challenge is to
take a file full of scraped product listings, each having only a price and a
title, and match them up with the known products.

By combining multiple, imperfect methods, we hope to achieve better preformance
than any of them individually.

We use a bayesian classifier to rank the best product choices for each listing.
Then we use a heuristic to pick the best one from this short list. Finally,
once we have grouped the listings by product, we filter out ones which have
prices that are too good to be true.


