docker build -t nanotoms . && docker run --rm -p 8000:80 -v $(pwd)/.:/app nanotoms
