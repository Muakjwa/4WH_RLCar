# AVRL (4Wheel Steering Autonomous Vehicle based on Reinforcement Learning)

This project is performed in Engineering Desing (MECH404)

Hardware : 4Wheel Steering Car
Software : Autonomous Driving

1. 4Wheel Steering Car observes environment by 4 cameras.
2. PPO model(RL) decides action based on 4 distance to line from car.
3. 4Wheel Steering Car do action RL model decides.
(Action mode is only composed by [Go Straight, Go Diagonal, Turn])

## Hardware

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/985e7e16-044a-4b1c-9281-7709baf447b4" align="center" width="70%">  
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/100abb80-0205-4559-9080-1434efc75999" align="center" width="70%">  
    </td>
  </tr>
</table>

Hardware include individual wheel controled by each teensy.
RL model provides proper action to each teensy for one action.

Car used two Raspberry Pi to allocate tasks deciding actions & observing environments.
(Two Raspberry Pi connect each other by Ethernet communication.)

Deciding action Raspberry Pi send action command to each teensy.

## Software

Software consists of three parts.

1. vision process - 4 cameras get frames & distance from car to lines
2. simulation - 2D simulation to train RL autonomous vehicle model (openai gym)
3. sim2real - by using vision process & RL model, execute hardware properly

<td style="display: flex; justify-content: center;">
  <div align="center">
    <center><img src="https://github.com/user-attachments/assets/1927c6c4-2c11-4868-b1b4-25f7657f45d6" width="70%">  </center>
  </div>
</td>

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8ede34cd-045e-48cd-aa54-6f5f9ca5827c" width="200px" >
      <br>
      <em>Straight Mode</em>  
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8285c611-058e-4d9b-b465-f3f24b64fbaf" width="200px" > 
      <br>
      <em>Diagonal Mode</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ccff4a92-3cb8-4792-ac1c-9894d1f6eda8" width="200px" > 
      <br>
      <em>Zero Turn Mode</em> 
    </td>
  </tr>
</table>


