# CreditLens AI frontend: multi-stage build.
#
# Stage 1 builds the Vite production bundle with the API base
# URL baked in as a same-origin "/api" path (see App.jsx), so the
# browser never needs to know the backend container's hostname.
# Stage 2 serves that static bundle with nginx and proxies /api/*
# to the backend service.

FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
