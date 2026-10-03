# LOBBOT — Identity Lab

Brand exploration for the autonomous, capability-guided model-compression agent.

Serve `dist/` with any static HTTP server. `index.html` compares three identities; `landing.html` presents the proposed developer landing. Add `?direction=incision`, `?direction=core` or `?direction=signal` to preview an identity.

The process animation and YAML-like contract are illustrative. This project does not perform model compression.

Assets live in `dist/assets/logos/`. All SVG wordmarks are outlined and need no installed fonts. Font licences accompany the local fonts. The ZIP includes the logos, brand guidelines and fonts.

`generate-assets.py` reproduces the SVGs and WOFF2 files using fonttools and Brotli from the supplied TTFs. The website has no runtime dependencies or build step.
