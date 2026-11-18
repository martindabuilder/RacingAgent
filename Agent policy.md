PPO agent with an MLP neuron network

The agent will be penalized if any of the following actions occur:
Moving before the 5 lights have gone out (false start);
Not moving at all after the 5 lights have went out;
Moving too slowly;
Not moving at all/stopping completely.
Going outside of the map causes an immidiate end to the lap and reset to the position.
Track limit violations
Not activating the DRS in the DRS zone;
Not deactivating the DRS after the end of the DRS zone;
Reversing on the grid.

Specific track limits penalties:
Going off with one tire for more than 2.5 seconds will result in a penalty;
Going off for for or less than 2 seconds with 2 same side wheels will not result in penalty;
Going off with 2 wheels on the side being outside of the track for more than 2 seconds will warrant a penalty;
Going off with all 4 wheels but still being on the track will result in a harsher penalty;
Going off with all 4 wheels and reaching the end of the map results in immidiate episode end.


Things to fix:
Holding the brakes for infinity gives it points.

Parameters:
max speed: 90m/s
max lateral acc: 16m/s^2
minimum speed target 15m/s
braking deceleration 12m/s^2
maximum stall time 2s
minimum stall speed 0.5m/s


Rewards: 
progress delta * 0.12 - progress on track for the current step, 0.12 for each meter
speed reward = 0.10 * speed ratio, 0.10 every step
corner entry reward = 0, braking reward = 0
When going off the track, progress/speed rewards dpnt count up anymore, while the punishments do count.


Penalties:
overspeed: overspeed ratio ** 2 0.08, 0.25
cmp target / pen
100%	     0.000
110%	     0.0008
125%	     0.002
150%	     0.005
200%	     0.032
300%	     0.128
400%	     0.25


lateral speed: (lateral speed / 25.0, 2.0) * 0.04 
(m/s)	penalty
0	    0.000
2	    0.0032
5	    0.008
10	    0.016
15	    0.024
25	    0.040
50+	    0.080


wrong direction/heading: (heading error / PI) * 0.015
heading angle / pen
0°	             0
15°	             0.00125
30°	             0.0025
45°	             0.00375
90°	             0.0075
180°	         0.015


sharp steering change: steering change * 0.005
steering change / pen
0.0	                0
0.2	                0.001
0.4	                0.002
0.6	                0.003
0.8	                0.004
1.0	                0.005
2.0	                0.010


unnecessary braking:
less than 2 m/s: 0.025
speed 90% < target: 0.015 x speed deficit
speed between 90 and 105%: 0.005
over 105% 0


stalling
stall time: 2s, and less than 0.5m/s
0.10 each step