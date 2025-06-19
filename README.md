# RLVF

RLVF is a library for reinforcement learning for verified rewards.

Let's start with chess puzzles. We can use the huggingface chess dataset: `Lichess/chess-puzzles`. One row of the dataset looks like this:

```json
{
  "PuzzleId": "0009B",
  "FEN": "r2qr1k1/b1p2ppp/pp4n1/P1P1p3/4P1n1/B2P2Pb/3NBP1P/RN1QR1K1 b - - 1 16",
  "Moves": "b6c5 e2g4 h3g4 d1g4",
  "Rating": 1112,
  "RatingDeviation": 74,
  "Popularity": 87,
  "NbPlays": 569,
  "Themes": "advantage middlegame short",
  "GameUrl": "https://lichess.org/4MWQCxQ6/black#31",
  "OpeningTags": "Kings_Pawn_Game Kings_Pawn_Game_Leonardis_Variation"
}
```

The model gets it correct if they match `Moves` exactly. You should use the `verifiers` library for the heavy lifting.
