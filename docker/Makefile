clean:
	find . -name '__pycache__' -type d -exec rm -r {} +

build-image:
	docker build -t=siamese-reid:v1 .

run:
	docker run -it -v /Users/3i-a1-2021-15/Developer/projects/datasets/Market-1501-v15.09.15:/home/app/datasets/Market-1501-v15.09.15 siamese-reid:v0

