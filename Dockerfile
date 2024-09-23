FROM node:16

WORKDIR /app

COPY . .

RUN git clone https://github.com/ryohey/signal.git /tmp/signal

# Compare the contents of the local folder and the cloned repo
# If the repository has newer content, pull the latest changes
RUN bash -c 'diff -rq /app /tmp/signal || cp -r /tmp/signal/* /app'

RUN npm install

RUN npm run build

EXPOSE 3000

# Start the application
CMD ["npm", "start"]
