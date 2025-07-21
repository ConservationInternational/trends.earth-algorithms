def trans_factors_for_custom_legend(trans_factors, ipcc_nesting):
    """
    trans_factors is a list containing two lists, where each matched pair from
    trans_factors[0] and trans_factors[1] is a transition code (in first list), and then
    what that transition code should be recoded as (in second list)

    this function takes in SOC transition factors defined against the IPCC legend
    and spits out transition factors tied to a child legend of the IPCC legend
    """

    transitions = []
    to_values = []

    assert len(trans_factors[0]) == len(trans_factors[1])

    for n in range(0, len(trans_factors[0])):
        trans_code = trans_factors[0][n]
        value = trans_factors[1][n]
        ipcc_initial_class_code = int(trans_code / 10)
        ipcc_final_class_code = trans_code % 10
        custom_initial_codes = ipcc_nesting.nesting[ipcc_initial_class_code]
        custom_final_codes = ipcc_nesting.nesting[ipcc_final_class_code]
        for initial_code in custom_initial_codes:
            # Convert from class code to index, as index is used in
            # the transition map
            initial_index = ipcc_nesting.child.class_index(
                ipcc_nesting.child.class_by_code(initial_code)
            )
            for final_code in custom_final_codes:
                # Convert from class code to index, as index is used in
                # the transition map
                final_index = ipcc_nesting.child.class_index(
                    ipcc_nesting.child.class_by_code(final_code)
                )
                transitions.append(
                    initial_index * ipcc_nesting.get_multiplier() + final_index
                )
                to_values.append(value)

    return [transitions, to_values]
