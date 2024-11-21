FROM node:lts

WORKDIR /app

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm install

COPY . .

RUN npm run build

EXPOSE 3000

# Start the application
CMD ["npm", "run", "serve"]