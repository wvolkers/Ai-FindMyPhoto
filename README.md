<!-- This is based on the markdown template for the final project of the Building AI course, created by Reaktor Innovations and University of Helsinki -->

# Ai-Jigsawpuzzle-Buddy
## Summary

Ai-Jigsawpuzzle-Buddy will allow you to make a picture of your puzzle pieces along with the partly finished puzzle. Based on this picture it will generate an advice on placement of a next puzzle piece.

( Final project for the Building AI course )

## Background

You might get stuck while trying to complete a puzzle, none of the pieces seem to fit! You might ask a friend or a family member to help you. However, most people seem to really dislike your hobby. This is where Ai-Jigsawpuzzle-Buddy comes to the rescue!

## How is it used?

You make a picture of the puzzle pieces along with the partly finished puzzle. Ai-Jigsawpuzzle-Buddy will mark one of the unconnected pieces with a square en place a square where it might fit to a number of connected pieces.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Jigsaw_puzzle_solving_2.jpg/960px-Jigsaw_puzzle_solving_2.jpg" width="300">

Pseudo code:
```
def main():
   puzzle = pictue
   connected_piece=extractconnected(puzzle)
   unconnected_pieces=extractunconnected(puzzle)

   piece=None
   location=None
   for p in unconnected_pieces:
      piece=p
      location = try_fit(connected_piece, piece)
      if location:
         break

   if location:
      mark(picture, piece)
      mark(picture, location)

main()
```

## Data sources and AI methods
* object detection and image segmentation techniques
* 

## Challenges

## What next?

## Acknowledgments
* wikimedia
* similar: https://github.com/OmarMusayev/puzzlesolver


