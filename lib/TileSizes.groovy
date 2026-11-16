import groovy.json.JsonSlurper

/**
 * Tile set (tile size in pixels, resolution in µm/px) for each requested model.
 *
 * Every model is tiled at Mussel's recommended size and resolution
 * (assets/recommended_patch_sizes.json, generated from Mussel's MODEL_PATCH_SIZES
 * and MODEL_TARGET_MPP by bin/sync_patch_sizes.py) unless explicitly overridden.
 * Size, in order of precedence:
 *   1. params.featurize.model_patch_sizes[model]  per-model override
 *   2. params.tiling.patch_size                   one size for every model
 *   3. recommended size
 * Resolution likewise:
 *   1. params.featurize.model_mpps[model]
 *   2. params.tiling.mpp                          one resolution for every model
 *   3. the model's target MPP, else DEFAULT_MPP (0.5 µm/px, 20x)
 * A slide encoder uses its own entries: its patch encoder runs on that tile set.
 *
 * Models with the same size and resolution share a tile set. All tile sets come
 * from one tissue segmentation per slide (Mussel tessellate seg_config.patch_sizes,
 * plus seg_config.patch_mpps when the resolutions differ). The first model in
 * model_types sets the "primary" set, published under tiles/ as before; other
 * sets go to tiles_<size>px/, or tiles_<size>px_<mpp>mpp/ at a resolution other
 * than the primary set's.
 *
 * A model listed in params.featurize.grid_from_primary is not tessellated at its size:
 * its tile set is derived from the primary set's tiles (Mussel's derive_tile_grid, every
 * cell containing a primary tile centre), at the primary set's resolution. E.g. use
 * this for titan_slide, so its 512 px grid matches the one built from the 224 px tiles.
 *
 * Featurization asks Mussel to check each model's tiles (tile_size_check):
 * "error" for models at their recommended size and MPP, "warn" where the size
 * or MPP was overridden on purpose.
 */
class TileSizes {

    static final Double DEFAULT_MPP = 0.5d

    static Map<String, Integer> recommended(projectDir) {
        def sizes = asset(projectDir).sizes
        return sizes.collectEntries { k, v -> [(k.toString()): v as Integer] }
    }

    /** Resolution (µm/px) a model's tiles must be read at, where Mussel specifies one. */
    static Map<String, Double> targetMpp(projectDir) {
        def mpp = asset(projectDir).target_mpp ?: [:]
        return mpp.collectEntries { k, v -> [(k.toString()): v as Double] }
    }

    static Map asset(projectDir) {
        return new JsonSlurper().parse(new File("${projectDir}/assets/recommended_patch_sizes.json")) as Map
    }

    static boolean sameMpp(a, b) {
        return Math.abs((a as Double) - (b as Double)) < 1e-6d
    }

    /** µm/px as Mussel writes it in output paths ({mpp}, Python's :g): 0.5, 1, 0.25. */
    static String mppTag(mpp) {
        return new BigDecimal((mpp as Double).toString()).stripTrailingZeros().toPlainString()
    }

    /** Channel key of a tile set. */
    static String key(Map spec) {
        return "${spec.size}px@${mppTag(spec.mpp)}".toString()
    }

    /**
     * Mussel tile_size_check mode for a model tiled with `spec` ([size:, mpp:]):
     * params.featurize.tile_size_check (default "error") when size and MPP are the
     * recommended ones, "warn" when either was overridden on purpose, null to not
     * pass the option (Mussel without tile_size_check).
     */
    static String checkMode(params, projectDir, String model, Map spec) {
        def mode = params.featurize.tile_size_check
        if (mode == null) {
            return null
        }
        def rec = recommended(projectDir)[model]
        def target = targetMpp(projectDir)[model]
        boolean overridden = (rec != null && rec != spec.size) || (target != null && Math.abs(target - (spec.mpp as Double)) > 0.025d * target)
        return overridden ? "warn" : mode.toString()
    }

    /** Ordered map model -> [size: px, mpp: µm/px] for params.featurize.model_types. */
    static Map<String, Map> resolve(params, projectDir, log = null) {
        def models = params.featurize.model_types instanceof List
            ? params.featurize.model_types
            : [params.featurize.model_types]
        def rec = recommended(projectDir)
        def target = targetMpp(projectDir)
        def perModel = (params.featurize.model_patch_sizes ?: [:]).collectEntries { k, v -> [(k.toString()): v as Integer] }
        def perModelMpp = (params.featurize.model_mpps ?: [:]).collectEntries { k, v -> [(k.toString()): v as Double] }
        def global = params.tiling.patch_size != null ? params.tiling.patch_size as Integer : null
        def globalMpp = params.tiling.mpp != null ? params.tiling.mpp as Double : null
        def out = new LinkedHashMap<String, Map>()
        def overridden = []
        models.each { m ->
            def name = m.toString()
            Integer size
            if (perModel.containsKey(name)) {
                size = perModel[name]
                if (rec[name] != null && rec[name] != size) overridden << "${name}=${size} px (recommended ${rec[name]}, featurize.model_patch_sizes)"
            } else if (global != null) {
                size = global
                if (rec[name] != null && rec[name] != global) overridden << "${name}=${global} px (recommended ${rec[name]}, tiling.patch_size)"
            } else if (rec[name] != null) {
                size = rec[name]
            } else {
                throw new IllegalArgumentException(
                    "No recommended tile size for model '${name}'. Add it to assets/recommended_patch_sizes.json " +
                    "or set featurize.model_patch_sizes.${name}.")
            }
            Double mpp
            if (perModelMpp.containsKey(name)) {
                mpp = perModelMpp[name]
                if (target[name] != null && !sameMpp(target[name], mpp)) overridden << "${name} at ${mpp} µm/px (authors' ${target[name]}, featurize.model_mpps)"
            } else if (globalMpp != null) {
                mpp = globalMpp
                if (target[name] != null && !sameMpp(target[name], mpp)) overridden << "${name} at ${mpp} µm/px (authors' ${target[name]}, tiling.mpp)"
            } else {
                mpp = target[name] != null ? target[name] : DEFAULT_MPP
            }
            out[name] = [size: size, mpp: mpp]
        }
        if (overridden && log != null) {
            log.warn "Tiles overridden away from the model's recommended size/resolution: ${overridden.join('; ')}"
        }
        return out
    }

    /**
     * Distinct tile sets of `specs` (resolve()), primary (first model's) first:
     * [size:, mpp:, key:, dir: publish subdir, suffix: manifest key suffix,
     *  derived: built from the primary set's tiles rather than tessellated].
     * A set is derived when every model using it is in `derivedModels`
     * (params.featurize.grid_from_primary).
     */
    static List<Map> tileSets(Map<String, Map> specs, List derivedModels = []) {
        def unique = []
        specs.values().each { s -> if (!unique.any { it.size == s.size && sameMpp(it.mpp, s.mpp) }) unique << s }
        def primary = unique[0]
        def derivedNames = (derivedModels ?: []).collect { it.toString() }
        def unknown = derivedNames.findAll { !specs.containsKey(it) }
        if (unknown) {
            throw new IllegalArgumentException("featurize.grid_from_primary lists ${unknown}, which are not in featurize.model_types")
        }
        return unique.collect { s ->
            def suffix = s.is(primary) ? '' : (sameMpp(s.mpp, primary.mpp) ? "_${s.size}px" : "_${s.size}px_${mppTag(s.mpp)}mpp")
            def users = specs.findAll { m, spec -> spec.size == s.size && sameMpp(spec.mpp, s.mpp) }.keySet()
            boolean derived = !s.is(primary) && users.every { derivedNames.contains(it) }
            if (derived && !sameMpp(s.mpp, primary.mpp)) {
                throw new IllegalArgumentException(
                    "featurize.grid_from_primary: ${users} need ${mppTag(s.mpp)} µm/px but the primary tiles are at " +
                    "${mppTag(primary.mpp)} µm/px; a derived grid keeps the primary resolution.")
            }
            if (derived && s.size < primary.size) {
                throw new IllegalArgumentException(
                    "featurize.grid_from_primary: ${users} use ${s.size} px tiles, smaller than the primary ${primary.size} px.")
            }
            [size: s.size, mpp: s.mpp, key: key(s), dir: "tiles${suffix}".toString(), suffix: suffix.toString(), derived: derived]
        }
    }

    /** Models whose tile set is derived from the primary tiles (featurize.grid_from_primary). */
    static List derivedModels(params) {
        def v = params.featurize.grid_from_primary
        return v == null ? [] : (v instanceof List ? v : [v])
    }

    /** The tile set a model uses. */
    static Map tileSetFor(List<Map> tileSets, Map spec) {
        return tileSets.find { it.size == spec.size && sameMpp(it.mpp, spec.mpp) }
    }

    /**
     * Slide encoders the one-step workflow can't tile as `specs` (resolve()) asks,
     * one message each. There a slide encoder runs in Mussel's multi-model
     * tessellate_extract_features (model_type and slide_model_type both set), which
     * tiles it at its recommended size whatever seg_config.patch_size says, and takes
     * seg_config.mpp = 0.5 as unset, using the encoder's own resolution instead
     * (e.g. 1.0 µm/px for chief_slide). The two-step workflow tiles as asked.
     */
    static List<String> oneStepConflicts(Map<String, Map> specs, Collection slideEncoders, projectDir) {
        def rec = recommended(projectDir)
        def target = targetMpp(projectDir)
        def encoders = slideEncoders.collect { it.toString() }
        def out = []
        specs.each { m, spec ->
            if (!(m in encoders)) {
                return
            }
            def problems = []
            if (rec[m] != null && rec[m] != spec.size) {
                problems << "${spec.size} px (Mussel ignores seg_config.patch_size here and uses ${rec[m]} px)"
            }
            def own = target[m] != null ? target[m] : DEFAULT_MPP
            if (sameMpp(spec.mpp, DEFAULT_MPP) && !sameMpp(own, DEFAULT_MPP)) {
                problems << "${mppTag(spec.mpp)} µm/px (Mussel takes seg_config.mpp=0.5 as unset here and uses ${mppTag(own)} µm/px)"
            }
            if (problems) {
                out << "${m} at ${problems.join(' and ')}".toString()
            }
        }
        return out
    }

    /** True when the tile sets need more than one resolution (Mussel seg_config.patch_mpps). */
    static boolean multiMpp(List<Map> tileSets) {
        return tileSets.any { !sameMpp(it.mpp, tileSets[0].mpp) }
    }
}
