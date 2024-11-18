FROM node:18

WORKDIR /app

COPY . .

RUN npm install && npm run build

EXPOSE 3000

# Start the application
CMD ["npm", "run", "serve"]