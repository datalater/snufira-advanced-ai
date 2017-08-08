
ⓒ JMC 2017

**고급인공지능, 노영균 교수님**  
\- nohyung@snu.ac.kr

---

+ 과학혁명

p.11
+ 모델링이란?
  + possible한 hypotehsis set을 만든다.
  + ex. linear model로만 한정시킨다.

+ 학습(learning)이란?
  + 여러 f중에서 하나의 f를 결정하는 것
  + 가장 적합한 theta를 가진 하나의 f를 결정하는 것

---

p.13 슬라이드에 나와 있는 종류가 중요한 게 아니라 순서의 의미를 아는 것이 중요하다.

+ 위로 갈수록 discriminative하고, 아래로 갈수록 generative한 모델이다.
+ discriminative 알고리즘이 만드는 H와
+ generative 알고리즘이 만드는 H는 상이하다.
+ 상황마다 각각 다른 망치가 필요하듯, 데이터셋에 딱 맞는 알고리즘을 사용하는 것이 중요하다.
+ 위염약을 먹고 나으면 위염임을 확진한다.
+ 데이터를 받았으면 이 데이터가 어떤 모양으로 분포할지 가정을 한 다음에, 그 가정에 따른 분포를 가장 잘 분리하는 알고리즘을 돌려봐야 안다.
+ 알고리즘을 돌렸는데 마음에 안 든다, 더 좋은 알고리즘이 있을 것 같다면, 처음에 세운 가정을 수정하고, 수정된 가정에 따라서 분포하는 데이터를 가장 잘 분류하는 알고리즘을 사용해본다.
+ 데이터가 오면 discriminative로 돌려보고 non-discriminative로 돌려본다.
+ 만약 logistic regression으로 했는데 잘 안되서 Artificial Neural Network를 선택하는 것은 옳지 않다.
+ discriminative에서 잘 안되었으면 또 다른 discriminative에서도 잘 안될 확률이 높다.
+ 그 다음 순서로는 generative한 모델을 선택하는 것이 옳다.


p.13

+ generative : 데이터가 많아지면 잘 못한다.
+ discriminative : 데이터가 적으면 잘 못한다.
+ 데이터가 많으면 정보가 많으므로 정밀하게 fitting하는게 더 잘하는 것이므로 discriminative가 좋은 것이다.
+ 데이터가 적으면 쓸데 없는 정보를 많이 버리기 때문에 상대적으로 잘 fitting하므로 generative가 잘한다.
+ 데이터가 적으면 discriminative는 noise에 휩쓸린다.

p.13

+ 데이터의 많고 적음의 기준은?
+ discriminative 모델을 돌렸을 때 generative 보다 잘 fitting하면 그러면 데이터가 많이 있다고 봐야 한다.

---

p.14

+ 기계학습을 학문적으로 이해하는 방법
+ ex. 알고리즘의 관계를 알아야 한다. discriminative와 generative의 차이점.

---
