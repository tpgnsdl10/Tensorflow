import tensorflow as tf #tensorflow 사용하기위하여 import
from tensorflow.examples.tutorials.mnist import input_data

## Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

## Set up model
x = tf.placeholder(tf.float32, [None, 784]) #심볼릭 변수 사용하여 상호작용하는 작업 기술
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))#가중치와 편향값 사용 위하여 모델 파라미터 Variable 사용. tf.Variable 사용하여 초기값 생성
y = tf.nn.softmax(tf.matmul(x, W) + b) #모델 구현.tf.matmul(x, W) 표현식으로 xx 와 WW를 곱한 다음 b를 더하고 tf.nn.softmax 를 적용.

y_ = tf.placeholder(tf.float32, [None, 10]) #교차 엔트로피를 구현하기 위해 정답을 입력하기위해 우선적으로 새 placeholder 추가.

cross_entropy = -tf.reduce_sum(y_*tf.log(y))#교차 엔트로피 구현. tf.log는 y의 각 원소의 로그값 계산 후 y_ 의 각 원소들에 각각에 해당되는 tf.log(y)를 곱하고 tf.reduce_sum은 텐서의 모든 원소를 더함.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
init = tf.initialize_all_variables()#실행 전 마지막으로 만든 변수들을 초기화하는 작업을 추가.

sess = tf.Session()
sess.run(init) #세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행.

# Learning
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #학습 1000번 실행. 각 반복 단계마다 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옴. placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행.



# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#tf.argmax(y,1) 는 진짜 라벨이 tf.argmax(y_,1) 일때 모델이 각 입력에 대하여 가장 정확하다고 생각하는 라벨. tf.equal 을 이용해 예측이 실제와 맞았는지 확인할 수 있음.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))#테스트 데이터를 대상으로 정확도를 확인.
