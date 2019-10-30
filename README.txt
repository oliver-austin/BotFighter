This is our repository for building an agent using Deep Q-Learning to play Street Fighter II

Add info here as necessary

git pull - updates all branches
git checkout 'name of new branch' - moves you to new branch
git pull - pull the branch your looking at

Library requirements can be installed from the requirements.txt file

To customize the reward function, the scenario.json file in the retro library files must be changed.
The default value gives a reward based on the player's score.
To reward based on the difference in player's health, paste the following in the reward>variables object in the
scenario file located at: site-packages/retro/data/stable/stable/StreetFighterIISpecialChampionEdition-Genesis/scenario.json

      "health": {
        "reward": 1.0
      },
      "enemy_health": {
        "reward": -1.0
      }
