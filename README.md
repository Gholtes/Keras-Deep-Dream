# Keras-Deep-Dream
An implementation of Deep Dream in Keras, with examples. A bare bones version is also included for ease of understanding.

Method:

1) Load and preprocess image. Save 2 more copies.
2) Select a layer(s) in the network. The sum of this layer (the activations at this depth in the network) is to be maximised. 
   The loss is the negtaive of this sum.
3) Shrink the image down, and one of the copies.
4) For successivly larger image sizes:

4.1)Enlarge the input image and copy.

4.2)Shrink the second copy down to the current size.
  
4.3)The difference between the two copies is the information lost through enlarging the image.

4.4)Add this lost information back to the input image

4.5)For a few interations:

4.5.1) Get the gradient of the loss with respect to the input image.

4.5.2)Perform gradient accent by adding this gradient (multiplied by a constant) to the input image.

5) Resize the image back to its original dimensions, adding back any final lost information
6) Convert to RBG and save
