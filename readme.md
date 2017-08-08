
ⓒ JMC 2017

**고급인공지능, 노영균 교수님**  
\- nohyung@snu.ac.kr

---

## Summary

### p.11 Learning from Data for "Prediction"

"Learning : 가장 잘 fitting하는 함수 $f$의 $\theta$를 집어내는 것"

+ train data로부터 prediction을 위한 함수 $f$를 만들어야 한다.
+ 함수 $f(x;\theta)$는 input variable $x$와 parameter $\theta$로 구성된다.
+ train data를 만족시키는 함수 $f$는 여러 개 존재할 수 있다.
+ 연구자는 주어진 데이터를 보고 가정을 한 후 함수 $f$에 대한 모델 셋($H$)을 한정시켜야 한다.
+ 가장 잘 fitting하는 함수 $f$를 도출하기 위해 하나의 $\theta$를 집어낸다.
+ 함수 $f$의 가장 적합한 parameter $\theta$를 알아내는 것을 학습(learning)이라 한다.

> **Note:** (1) 학습 = 모델 set을 한정시킨다 = 함수 set을 한정시킨다 = 모델의 차수를 낮춘다 = parameter의 수를 줄인다 | (2) 넓은 범위에서는 학습과 prediction을 함께 묶어서 학습이라고도 한다.

### p.13 Classification Algorithms 순서의 의미

+ 위로 갈수록 discriminative한 모델이다.
+ 아래로 갈수록 generative한 모델이다.
+ discriminative vs. generative는 기계학습 알고리즘을 가장 잘 구별하는 기준이다.
+ 알고리즘의 특징을 잘 알고 data set에 맞는 성질의 알고리즘을 사용하는 것이 중요하다.


### p.13 Classification Algorithms :: discriminative vs. generative

#### fitting 방식

+ discriminative : 정밀하게 fitting 한다.
+ generative : 쓸데 없는 정보를 버리면서 fitting 한다.

#### 유리한 상황, 불리한 상황

+ discriminative : 데이터가 많을 때 유리하다. 데이터가 적으면 잘 못한다.
  + 데이터가 많으면 정보가 많으므로 정밀하게 fitting하는게 더 잘하는 것이므로 discriminative가 좋다.
  + 데이터가 적으면 discriminative는 noise에 휩쓸릴 수 있다.
+ generative : 데이터가 적을 때 유리하다. 데이터가 많으면 잘 못한다.
  + 데이터가 적으면 generative는 쓸데 없는 정보를 많이 버리기 때문에 상대적으로 더 잘 fitting한다.

#### 데이터 양에 대한 판단 기준

+ 데이터 양에 대한 기준은 상대적이므로, 알고리즘을 돌려봐야 안다.
+ discriminative 모델을 돌렸을 때 generative 보다 잘 fitting하면 데이터가 많이 있다고 봐야 한다.

---

#### extra :: 학습 방식

+ discriminative : class 간의 boundary를 학습(learn)한다.
+ generative : 각 class의 분포를 학습(learn)한 후 boundary를 결정한다.

> **Note:** 출처 : `https://stats.stackexchange.com/questions/12421/generative-vs-discriminative`

#### extra :: 수학 공식

+ discriminative : conditional probability `p(y|x)`
+ generative : joint probability distribution `p(x, y)`

> **Note:** 출처 : `https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm`

---

### p.13 주어진 데이터로부터 알고리즘을 선택하는 방법

+ 데이터가 어떤 모양으로 분포할지 가정을 한다.
+ 가정에 따른 분포를 가장 잘 분리할 것 같은 알고리즘을 임의로 선택하고 돌려본다.
+ 알고리즘을 돌렸는데 마음에 들지 않고 더 좋은 알고리즘이 있을 것 같다면, 처음에 세운 가정을 수정하고 수정한 가정에 따라서 분포하는 데이터를 가장 잘 분류하는 다른 알고리즘을 선택하고 돌려본다.
+ 예를 들어, discriminative한 알고리즘으로 돌렸는데 마음에 들지 않는다면 generative 알고리즘으로 돌려본다.
+ 가령 logistic regression으로 했는데 잘 안되서 Artificial Neural Network를 선택하는 것은 옳지 않다.
+ discriminative에서 잘 안되었으면 또 다른 discriminative에서도 잘 안될 확률이 높기 때문이다.

### p.15 학습을 위한 가정 (1/2)

+ train data와 test data는 같은 모수 data로부터 randomly 선택된다.
+ 즉, 같은 underlying density function(밀도함수)로부터 randomly 추출된다.
+ 같은 밀도함수로부터 추출되므로 regularity가 존재한다.

### p.15 예시 :: 두 가지 클래스의 분포에 대한 가정

+ 클래스 c1과 c2가 있다고 해보자.
+ 각 클래스의 점들은 같은 밀도함수로부터 추출되었다.
+ 그러나 각 클래스는 서로 다른 parameter를 가진다.

### p.15 예시 :: Bayes classification

 + resume @@@

### p.17 Quantify the Evaluation

+

----

## Lecture note

+ 과학혁명

---

p.11

+ 모델링이란?
  + f(x;theta)를 만족시키는 hypothesis set을 무엇으로 할지 결정한다.
  + possible한 hypotehsis set을 만든다.
  + ex. linear model로만 한정시킨다.

+ 학습(learning)이란?
  + 여러 f중에서 하나의 f를 결정하는 것
  + f(x;theta), Learn theta
  + 가장 적합한 theta를 가진 하나의 f를 결정하는 것

---

p.13 슬라이드에 나와 있는 알고리즘의 종류가 중요한 게 아니라 '**순서의 의미**'를 아는 것이 중요하다.

+ 위로 갈수록 discriminative하고, 아래로 갈수록 generative한 모델이다.
+ discriminative 알고리즘이 만드는 H와
+ generative 알고리즘이 만드는 H는 상이하다.

p.13 슬라이드 속 그림의 의미

+ 상황마다 각각 다른 망치가 필요하듯, 데이터셋에 딱 맞는 알고리즘을 사용하는 것이 중요하다.

p.13 의사가 위염을 진단하는 방법은 모델을 선택하는 방법과 유사하다.

+ 환자가 위염약을 먹고 나으면 위염임을 확진한다.
+ 데이터를 받았으면 이 데이터가 어떤 모양으로 분포할지 가정을 한 다음에, 그 가정에 따른 분포를 가장 잘 분리하는 알고리즘을 돌려봐야 안다.
+ 알고리즘을 돌렸는데 마음에 안 든다, 더 좋은 알고리즘이 있을 것 같다면, 처음에 세운 가정을 수정하고, 수정된 가정에 따라서 분포하는 데이터를 가장 잘 분류하는 알고리즘을 사용해본다.
+ 데이터가 오면 discriminative로 돌려보고 non-discriminative로 돌려본다.
+ 만약 logistic regression으로 했는데 잘 안되서 Artificial Neural Network를 선택하는 것은 옳지 않다.
+ discriminative에서 잘 안되었으면 또 다른 discriminative에서도 잘 안될 확률이 높다.
+ 그 다음 순서로는 generative한 모델을 선택하는 것이 옳다.

p.13 generative vs. discriminative

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

p.15

+ Bayes classification
+ 에러가 가장 작은 classification이다.
+ 그러나 우리가 class1에 대한 분포인 p1과 class2에 대한 분포인 p2를 알지 못하므로 사용할 수 없다.

---

p.16~18 넘어야 되는 산

+ 데이터셋을 어떻게 만져야 또는 모델링을 어떻게 해야 L(f)와 L^(f)의 차이를 줄일 수 있을까 고민하는 것이 머신러닝의 핵심이다.
+ learnability가 성립되는 모델 : 우리가 사용할 수 있는 모델 = consistent model

---

p.22~23

+ 인간의 직관에는 편향이 있기 때문이다.
+ 데이터에 대한 모든 모델은 가정이다.
+ finite한 데이터를 보고 predict하려면 모델(가정)이 필요하다.
+ generalization : 주어진 정보를 가지고 전체에 대해서 어떤 점의 class가 어떻게 될 것인지 generalize한다.
+ 모델이 있어야 generalize가 가능하다.
+ 하지만 우리는 모든 모델을 사용하지는 않는다.
+ 가능한 모델을 한정(confine)시켜야 한다. = 함수 set을 한정시킨다. (우리의 가정에 근거하여) = 모델의 차수를 낮춘다 = parameter의 개수를 한정시킨다.
+ 한정시키는 경우에 따라 모델이 learnability 만족할 수도 있고 아닐 수도 있다.


p.24

+ training data에 있는 모든 정보에 fitting 하면 noise 정보에도 fit하게 되고 그러면 test data는 제대로 predict하지 못하게 된다. 이렇게 train error가 지나치게 낮은 모델을 overfitting이라고 한다.


p.26

+ regularization : 모델을 confine시키는 것과 정확히 연결된다.
+ 이 슬라이드에 있는 내용은 p.24~25에 나와있는 그래프와 연결되어 있다.

p.28

+ bin(x) = x가 속하는 그리드 영역

p.29

+ classifier1의 장단점
  + 장점 : 데이터가 무진장 많아지면 baeys error에 근접한다. 왜냐하면 p1과 p2로 분리하는 것과 방법이 같아지기 때문.
  + 단점 : 그리드가 디멘젼이 커지면 exponential하게 늘어난다. 사실상 계산이 불가능해진다. 디멘젼이 높아질수록 그리드를 채울 수 있는 데이터가 급격하게 부족해진다.


p.30

+ classifier2의 장단점
  + 장점 : 계산이 쉽다. 디멘젼이 커져도 문제가 없다.
  + 단점 : 데이터가 아무리 많아도 baeys error에 근접할 수 없다.
    + underlying density function의 boundary가 1차원으로 나눠질 수 없다. (?)

p.33

+ 거리가 제일 가까운 5개의 점을 뽑은 후 다수결
  + 장점 : 데이터가 많아지면 bayes error
  + 단점 : 디멘젼이 높아져도 상관이 없다. 데이터가 적을 경우 매우 불확실해진다. 디멘젼이 커지면 계산은 쉽지만 디멘젼을 채울 데이터가 부족해질 수 있다.

p.34

+ training data에 대한 100% 적중하는 classifier4
  + 단점 : test data를 맞추지 못한다. (overfitting)

p.35

  + 가우시안 classifier4 (generative model)
    + 장점 :
    + 단점 : bayes error 근접 불가. 디멘젼이 높아지면 parameter 수가 엄청나게 늘어나서 overfitting될 수 있다.

---
