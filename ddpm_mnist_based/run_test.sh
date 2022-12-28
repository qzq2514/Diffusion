
# random_gen
CUDA_VISIBLE_DEVICES=7 python test_solver.py \
                              --model_path "./train_results/StyleGAN_face128/StyleGAN_face_size128_225.pt" \
                              --image_size 128 \
                              --image_channel 3 \
                              --sample_num 25 \
                              --sample_timestep_num 1000 \
                               --test_model "random_gen"


# interpolate

#CUDA_VISIBLE_DEVICES=6 python test_solver.py \
#                              --model_path "./train_results/cifar10/cifar10_latest.pt" \
#                              --image_size 32 \
#                              --image_channel 3 \
#                              --sample_num 8 \
#                              --sample_timestep_num 1000 \
#                              --data_name "Cifar10" \
#                              --data_path "../dataset/cifar-10-batches-py" \
#                              --test_model "interpolate"

#CUDA_VISIBLE_DEVICES=6 python test_solver.py \
#                              --model_path "./train_results/fashion_mnist/fashion_mnist_latest.pt" \
#                              --image_size 32 \
#                              --image_channel 1 \
#                              --sample_num 10 \
#                              --sample_timestep_num 1000 \
#                              --data_name "Fashion_Mnist" \
#                              --data_path "../dataset/FashionMNIST/processed" \
#                              --test_model "interpolate"

#CUDA_VISIBLE_DEVICES=6 python test_solver.py \
#                              --model_path "./train_results/mnist/mnist_latest.pt" \
#                              --image_size 32 \
#                              --image_channel 1 \
#                              --sample_num 10 \
#                              --sample_timestep_num 1000 \
#                              --data_name "Mnist" \
#                              --data_path "../dataset/mnist.pkl.gz" \
#                              --test_model "interpolate"

#CUDA_VISIBLE_DEVICES=6 python test_solver.py \
#                              --model_path "./train_results/Flower102_64Size/flower102_size64_858.pt" \
#                              --image_size 64 \
#                              --image_channel 3 \
#                              --sample_num 7 \
#                              --sample_timestep_num 1000 \
#                              --data_name "Flower102" \
#                              --data_path "../dataset/Flower102" \
#                              --test_model "interpolate"

# interpolate
#CUDA_VISIBLE_DEVICES=6 python test_solver.py \
#                              --model_path "./train_results/StyleGAN_face128/StyleGAN_face_size128_225.pt" \
#                              --image_size 128 \
#                              --image_channel 3 \
#                              --sample_num 5 \
#                              --sample_timestep_num 1000 \
#                              --data_name "StyleGAN_Face" \
#                              --data_path "../dataset/StyleGAN_face" \
#                              --test_model "interpolate"

#11.213.122.6 hyRP3XZQ,
