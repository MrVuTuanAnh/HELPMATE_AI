const webpack = require('webpack');

module.exports = {
    webpack: {
        configure: (webpackConfig) => {
            const fallback = webpackConfig.resolve.fallback || {};
            Object.assign(fallback, { 
                "buffer": require.resolve("buffer/") 
            })
            webpackConfig.resolve.fallback = fallback;
            webpackConfig.plugins = (webpackConfig.plugins || []).concat([
                new webpack.ProvidePlugin({
                    Buffer: ['buffer', 'Buffer'],
                }),
            ]);
            return webpackConfig;
        }
    },
};
