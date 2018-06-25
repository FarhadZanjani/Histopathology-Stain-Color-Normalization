import tensorflow as tf

def GMM_M_Step(X, Gama, ClusterNo, name='GMM_Statistics', **kwargs):

    D, a, b = tf.split(X, [1,1,1], axis=3)
    
    WXd = tf.multiply(Gama, tf.tile(D ,[1,1,1,ClusterNo]))
    WXa = tf.multiply(Gama, tf.tile(a ,[1,1,1,ClusterNo]))
    WXb = tf.multiply(Gama, tf.tile(b ,[1,1,1,ClusterNo]))
    
    S = tf.reduce_sum(tf.reduce_sum(Gama, axis=1), axis=1)
    S = tf.add(S, tf.contrib.keras.backend.epsilon())
    S = tf.reshape(S,[1, ClusterNo])
    
    M_d = tf.div(tf.reduce_sum(tf.reduce_sum(WXd, axis=1), axis=1) , S)
    M_a = tf.div(tf.reduce_sum(tf.reduce_sum(WXa, axis=1), axis=1) , S)
    M_b = tf.div(tf.reduce_sum(tf.reduce_sum(WXb, axis=1), axis=1) , S)
    
    Mu = tf.split(tf.concat([M_d, M_a, M_b],axis=0), ClusterNo, 1)  
    
    Norm_d = tf.squared_difference(D, tf.reshape(M_d,[1, ClusterNo]))
    Norm_a = tf.squared_difference(a, tf.reshape(M_a,[1, ClusterNo]))
    Norm_b = tf.squared_difference(b, tf.reshape(M_b,[1, ClusterNo]))
    
    WSd = tf.multiply(Gama, Norm_d)
    WSa = tf.multiply(Gama, Norm_a)
    WSb = tf.multiply(Gama, Norm_b)
    
    S_d = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSd, axis=1), axis=1) , S))
    S_a = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSa, axis=1), axis=1) , S))
    S_b = tf.sqrt(tf.div(tf.reduce_sum(tf.reduce_sum(WSb, axis=1), axis=1) , S))
    
    Std = tf.split(tf.concat([S_d, S_a, S_b],axis=0), ClusterNo, 1)  
    
    dist = list()
    for k in range(0, ClusterNo):
        dist.append(tf.contrib.distributions.MultivariateNormalDiag(tf.reshape(Mu[k],[1,3]), tf.reshape(Std[k],[1,3])))
    
    PI = tf.split(Gama, ClusterNo, axis=3) 
    Prob0 = list()
    for k in range(0, ClusterNo):
        Prob0.append(tf.multiply(tf.squeeze(dist[k].prob(X)), tf.squeeze(PI[k])))
        
    Prob = tf.convert_to_tensor(Prob0, dtype=tf.float32)    
    Prob = tf.minimum(tf.add(tf.reduce_sum(Prob, axis=0), tf.contrib.keras.backend.epsilon()), tf.constant(1.0, tf.float32))
    Log_Prob = tf.negative(tf.log(Prob))
    Log_Likelihood = tf.reduce_mean(Log_Prob)
    
    return Log_Likelihood, Mu, Std
        