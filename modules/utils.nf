/**
 * Shared utility functions for the mussel-nf pipeline.
 *
 * Import with:
 *     include { resolvePrecision } from './utils'
 */

/**
 * Resolve the storage precision for a given model type.
 *
 * Checks params.featurize.model_precision_overrides first; falls back to
 * params.featurize.embedding_precision, then defaults to 'float32'.
 */
def resolvePrecision(model_type) {
    (params.featurize.model_precision_overrides && params.featurize.model_precision_overrides[model_type])
        ? params.featurize.model_precision_overrides[model_type]
        : (params.featurize.embedding_precision ?: 'float32')
}
