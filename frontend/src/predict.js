function predictTreeNode(nodeIndex, features, tree) {
  if (tree.childrenLeft[nodeIndex] === -1) {
    return tree.value[nodeIndex]
  }
  const featureIndex = tree.feature[nodeIndex]
  const threshold = tree.threshold[nodeIndex]
  if (features[featureIndex] <= threshold) {
    return predictTreeNode(tree.childrenLeft[nodeIndex], features, tree)
  } else {
    return predictTreeNode(tree.childrenRight[nodeIndex], features, tree)
  }
}

export function predictTariff(input, artifact) {
  const { model, features } = artifact
  const transformed = []

  for (const feature of features.numeric) {
    const rawValue = input[feature.name]
    const value =
      rawValue === null || rawValue === undefined || rawValue === ''
        ? feature.median
        : Number(rawValue)

    if (!Number.isFinite(value)) {
      throw new TypeError(`${feature.name} must be a valid number.`)
    }

    transformed.push((value - feature.mean) / feature.scale)
  }

  for (const feature of features.categorical) {
    const rawValue = input[feature.name]
    const value =
      rawValue === null || rawValue === undefined || rawValue === ''
        ? feature.imputeValue
        : String(rawValue)

    for (const category of feature.categories) {
      transformed.push(category === value ? 1 : 0)
    }
  }

  if (model.type === 'linear' || model.coefficients) {
    if (transformed.length !== model.coefficients.length) {
      throw new Error('Model feature data does not match its coefficient count.')
    }
    return transformed.reduce(
      (prediction, value, index) =>
        prediction + value * model.coefficients[index],
      model.intercept,
    )
  }

  if (model.type === 'tree' || model.tree) {
    return predictTreeNode(0, transformed, model.tree)
  }

  if (model.type === 'forest' || model.trees) {
    let sum = 0
    for (const tree of model.trees) {
      sum += predictTreeNode(0, transformed, tree)
    }
    return sum / model.trees.length
  }

  if (model.type === 'gradient_boosting' || (model.trees && model.learningRate !== undefined)) {
    let val = model.initValue || 0
    for (const tree of model.trees) {
      val += model.learningRate * predictTreeNode(0, transformed, tree)
    }
    return val
  }

  throw new Error(`Unsupported model architecture in model artifact.`)
}
