# Taiwan baseball statical competition 2026

## origin

In 2026, MLB introduced a whole new challenging system, ABS (Automatic Ball-Strike Challenge) system into regular season.

This system provides 0.05%-win possibility change per game in regular season.
A player with excellent challenge skill will provide around .25 to .4 war (A war usually roughly represents 10 runs/1 win) value to the team, and a success challenge call could bring around .025 war. Looking back on the recent year's free agent (Free Agent) value, teams might need to pay around 8 million USD under current market. In other words, the value of a success challenge usually provides 0.2 million financial values, and an excellent challenger brings extra 2 to 4 million values to the team.

## Structure design

A challenge model will include 2 parts, a ball-strike prediction system, and a system to consider would the value of success challenge times the possibility overcomes the risk of lost challenge opportunity.

According to MLB official instruction players couldn't less too long to make the challenge decision and wait for any call from the clubhouse. So, the output of the challenge model should be a readable memo or some quick oral instructions from coaches between innings. 

simplify formula
EV = P_{success} \times \Delta WPA_{overturn} - (1 - P_{success}) \times OppCost_{WPA}

## Limitation

The current model does not explicitly model player-specific pitch recognition or pitch decision-making ability.
I decided not to deep into certain player due to the difficulty on valuating the pitch type and ball strike ability of each player, otherwise, the engage of similar research will lead to a high complexity model.

So, roughly, this system may be more suitable to catchers rather than a batter, a catcher knows what he did, for example, he called a outside, low slider, not a high fastball, he can tell if the pitch executes right or not.
To batter, we need to develop a new module or introduce the ability of a hitter distribution ability of pitch type, even the most fundamental thing, a fastball, a breaking ball, or an off-speed in the future, and coaches could give hitters oral guidelines, such as 'don't challenge anyway', or 'If it's a inner, high off-speed, and you're confused, directly challenge it without permission'.

# 