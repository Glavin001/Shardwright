# Self-contained fracture GLB, version 1

`SHARDWRIGHT_fracture` is a project/vendor extension, not a ratified Khronos
extension or registered vendor prefix. It packages the current FRAC schema in
standard GLB JSON + BIN chunks. It introduces no custom GLB chunk type.

Root JSON:

```json
{"extensionsUsed":["SHARDWRIGHT_fracture"],"extensions":{"SHARDWRIGHT_fracture":{
  "version":1,"encoding":"flatbuffers","physicsSchemaVersion":1,"bufferView":12,
  "coordinates":{"units":"meters","up":"Y","handedness":"right"},
  "materialLibrary":{"library_id":"...","library_version":"...","material":[],"interface_material":[]}
}}}
```

The referenced bufferView is an eight-byte-aligned slice of embedded buffer 0,
with no graphics target. It contains the original `.fracphys` bytes, including
FRAC file identifier. See `schemas/frac.fbs` for the complete versioned schema.
All rendering buffers/images must be embedded. The material library is a full
serialized `frac_material::MaterialLibrary`, including numeric physical properties;
fragment material IDs index `material`. Interface materials share the same ID
space: `interface_material[id - material.length]`. A bond interface material ID
of -1 means no specific interface material. Missing
optional material properties remain missing: consumers must choose/document their
constitutive model rather than interpreting null as zero strength. Bond strength
scale multiplies material capacity; it is not a breaking force in newtons.

Coordinates and hull vertices are in asset space, SI units. Inertia is a row-major
3x3 tensor about the fragment COM. Render meshes are COM-relative and node
translations are parent-relative. `Fragment.gltf_node` links into the same GLB;
`nodes[index].extras.fragment_id` must match. Select one active hierarchy frontier;
do not instantiate overlapping parent and child fragments. Render only that
frontier (the existing GLB contains geometry for every level).

The extension is listed in `extensionsUsed`, not `extensionsRequired`, because
ordinary viewers can render the geometry without physics. Destruction consumers
must explicitly require and validate it; generic glTF validators do not validate
its custom semantics. Optimizers/converters must preserve both the extension's
bufferView and the embedded fragment-to-node indices, or regenerate those indices.

Rust API: `write_fracture_glb(render_glb, fracphys, material_json)` and
`read_fracture_glb(glb)`; the latter returns the embedded bytes and library.
Use `frac_io::fb` for the complete typed FlatBuffers data. `read_physics` is only
a summary. Consumers validate extension/schema versions, buffer ranges, IDs,
finite mass properties and hull limits before allocating GPU resources.

CLI: `prefracture export-glb --input model.glb --out model.fracture.glb`.
The adjacent `.fracphys` is inferred; `--physics` overrides it. `--materials`
selects the same library used by the bake (default built-in). `--uncompressed`
expands meshopt data for minimal viewers. Packaging never runs fracture generation.

This is an asset, not a stress solver or a simulation cache. Fracture, constraints,
damage accumulation, topology updates and rendering remain the consuming runtime's
responsibility.
