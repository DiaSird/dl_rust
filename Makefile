default: run

run: train infer

train:
	python mnist_dl.py

infer:
	cargo run